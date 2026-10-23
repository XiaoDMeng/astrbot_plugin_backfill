"""
断联补读（astrbot_plugin_backfill）

问题：机器人掉线 / 重启 / NapCat 重连期间，用户在群里说的话机器人根本没收到，
这些消息不会进对话历史，重连后模型完全不知道刚才发生过什么。

做法：在「新会话」或「断联后的第一条消息」时，向 OneBot 要一段聊天记录，
从最新一条往前翻，一直找到机器人自己上一次发言为止；
把中间「别人说的话」按时间顺序拼成一段临时上下文塞进本轮请求，
然后正常回复。

约束：
- 只补机器人上次发言之后的、别人发的消息（不补它自己说过的话）
- 读不到聊天记录 / 接口不支持 / 超时，一律静默跳过，不抛错、不影响正常回复
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import traceback
from pathlib import Path

from astrbot.api import logger
from astrbot.api.event import filter, AstrMessageEvent, MessageChain
from astrbot.api.message_components import Plain
from astrbot.api.star import Context, Star
from astrbot.core.config.astrbot_config import AstrBotConfig

try:
    from astrbot.core.agent.message import TextPart
except ImportError:  # 兼容没有 TextPart 的旧版
    TextPart = None

# 单条消息最多保留多少字（防长文占满上下文）
_PER_MSG_CHARS = 300
# 拉聊天记录的等待上限（秒）
_FETCH_TIMEOUT = 8.0
# 去重记录落盘位置（跟插件同目录，删掉它就等于清空去重）
_SEEN_FILE = Path(__file__).with_name("seen.json")
# 每个会话最多记住多少条 message_id
_SEEN_PER_SESSION = 400
# 最多保留多少个会话的去重记录
_SEEN_MAX_SESSIONS = 60


class SeenStore:
    """去重集合，落盘保存：重启后不会把补过的消息再补一遍。

    结构：{ "<umo>": ["message_id", ...] }，按写入顺序保留最近 N 条。
    """

    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, list[str]] = {}
        self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except Exception as e:
            logger.warning(f"[backfill] seen.json 读取失败，按空处理：{e}")
            return
        if not isinstance(raw, dict):
            return
        for umo, ids in raw.items():
            if isinstance(ids, list):
                cleaned = [str(i) for i in ids if i]
                if cleaned:
                    self.data[str(umo)] = cleaned[-_SEEN_PER_SESSION:]

    def ids(self, umo: str) -> set[str]:
        return set(self.data.get(umo, ()))

    def add(self, umo: str, ids: list[str]) -> None:
        bucket = self.data.setdefault(umo, [])
        known = set(bucket)
        for mid in ids:
            if mid and mid not in known:
                bucket.append(mid)
                known.add(mid)
        if len(bucket) > _SEEN_PER_SESSION:
            del bucket[: -_SEEN_PER_SESSION]

    def save(self) -> None:
        try:
            if len(self.data) > _SEEN_MAX_SESSIONS:
                keep = list(self.data.items())[-_SEEN_MAX_SESSIONS:]
                self.data = dict(keep)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(self.data, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            os.replace(tmp, self.path)
        except Exception as e:
            logger.warning(f"[backfill] seen.json 写入失败：{e}")


class Main(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.cfg = config
        self._boot_at = time.time()
        # umo -> 上一次真正处理请求的时间戳
        self._last_req: dict[str, float] = {}
        # umo -> 已经处理过的 message_id（落盘，重启不丢）
        self.seen = SeenStore(_SEEN_FILE)
        self._boot_task: asyncio.Task | None = None
        # 并发补读时保护 seen 的“读-改-写”，否则多会话同时写会互相覆盖
        self._seen_lock = asyncio.Lock()

    def _cfg_int(self, key: str, default: int, lo: int = 0) -> int:
        """配置项转 int，坏了就退回默认值（别让一个错的配置把整段扫会话搞崩）。"""
        try:
            v = int(self.cfg.get(key, default))
        except Exception:
            v = default
        return max(lo, v)

    # ------------------------------------------------------------------ hooks

    @filter.on_llm_request()
    async def backfill(self, event: AstrMessageEvent, request) -> None:
        """在 LLM 请求前判断是否要补读，并把结果注入本轮上下文。"""
        try:
            if not self.cfg.get("enabled", True):
                return

            umo = event.unified_msg_origin
            now = time.time()
            last = self._last_req.get(umo, 0.0)
            gap = now - (last or self._boot_at)
            is_first = last == 0.0
            self._last_req[umo] = now

            contexts = getattr(request, "contexts", None) or []
            is_new_session = len(contexts) == 0

            gap_seconds = self._cfg_int("gap_seconds", 600, 0)
            triggered = (
                is_first
                or gap >= gap_seconds
                or (is_new_session and self.cfg.get("on_new_session", True))
            )
            if not triggered:
                return
            logger.info(
                f"[backfill] 触发 {umo} first={is_first} gap={gap:.0f}s new={is_new_session}"
            )

            diag: list[str] = []
            block = await self._build_block(event, request, diag)
            if block:
                self._inject(request, block)
                logger.info(f"[backfill] 已补读 {umo}：{len(block)} 字")
            else:
                logger.info(f"[backfill] 未补 {umo}：{' | '.join(diag) or '无诊断'}")
        except Exception as e:  # 补读永远不能影响正常回复
            logger.warning(f"[backfill] 异常：{e} | {traceback.format_exc()}")

    @filter.command("backfill_test", alias={"补读自测"})
    async def backfill_test(self, event: AstrMessageEvent):
        """管理自测：立刻拉一次聊天记录，把诊断结果回给发命令的人。"""
        if not event.is_admin():
            yield event.plain_result("只有管理员能用这个自测命令。")
            return
        diag: list[str] = []
        try:
            block = await self._build_block(event, _EmptyRequest(), diag, record=False)
        except Exception as e:
            yield event.plain_result(f"自测异常：{e!r}\n· " + "\n· ".join(diag))
            return
        head = "\n".join(f"· {d}" for d in diag) or "· （无诊断）"
        if block:
            yield event.plain_result(
                f"✅ 能补读（{len(block)} 字）\n\n{block[:1200]}\n\n诊断：\n{head}"
            )
        else:
            yield event.plain_result(f"❌ 没读到可补内容\n\n诊断：\n{head}")

    # ------------------------------------------------------------- internals

    async def _build_block(
        self, event: AstrMessageEvent, request, diag=None, record: bool = True
    ) -> str:
        """从事件里取出会话信息，再交给 _collect 拼补读文本。"""
        reasons: list[str] = diag if diag is not None else []

        def note(msg: str) -> None:
            reasons.append(msg)
            logger.debug(f"[backfill] {msg}")

        try:
            platform = event.get_platform_name()
        except Exception:
            platform = ""
        if platform != "aiocqhttp":
            note(f"平台是 {platform}，不是 aiocqhttp，跳过")
            return ""

        bot = getattr(event, "bot", None)
        if bot is None:
            note("拿不到 bot 实例（event.bot 为空）")
            return ""

        self_id = str(event.get_self_id() or "")
        group_id = str(event.get_group_id() or "")
        peer = "" if group_id else _private_peer(event, self_id)
        cur_id = str(getattr(getattr(event, "message_obj", None), "message_id", "") or "")
        cur_text = (event.message_str or "").strip()
        hours = self._cfg_int("hook_hours", 24, 0)
        since_ts = time.time() - hours * 3600 if hours > 0 else 0.0
        if since_ts:
            note(f"只看最近 {hours} 小时的消息")
        return await self._collect(
            bot=bot,
            self_id=self_id,
            group_id=group_id,
            peer=peer,
            umo=event.unified_msg_origin,
            known=_context_texts(request),
            diag=reasons,
            record=record,
            cur_id=cur_id,
            cur_text=cur_text,
            since_ts=since_ts,
        )

    async def _collect(
        self,
        *,
        bot,
        self_id: str,
        group_id: str,
        peer: str,
        umo: str,
        known: list[str],
        diag: list[str],
        record: bool = True,
        cur_id: str = "",
        cur_text: str = "",
        since_ts: float = 0.0,
        meta: dict | None = None,
    ) -> str:
        """拉聊天记录 → 找到机器人上次发言 → 拼出补读文本。"""

        def note(msg: str) -> None:
            diag.append(msg)
            logger.debug(f"[backfill] {msg}")

        if group_id and not self.cfg.get("include_group", True):
            note("群聊且未开启 include_group，跳过")
            return ""
        if not group_id and not self.cfg.get("include_private", True):
            note("私聊且未开启 include_private")
            return ""

        count = self._cfg_int("history_count", 50, 1)
        if group_id:
            action = "get_group_msg_history"
            params = {"group_id": int(group_id), "count": count}
        else:
            if not peer:
                note("私聊但认不出对方 QQ 号（可能取到的是自己）")
                return ""
            action = "get_friend_msg_history"
            params = {"user_id": int(peer), "count": count}

        try:
            payload = await asyncio.wait_for(
                bot.call_action(action, **params), timeout=_FETCH_TIMEOUT
            )
        except Exception as e:
            # 平台不支持该接口 / 超时 / 断开：按规则静默跳过
            note(f"{action} 调用失败：{e!r}")
            return ""

        msgs = _normalize(payload)
        if not msgs:
            note(f"{action} 返回空（payload 类型 {type(payload).__name__}）")
            return ""
        note(f"{action} 拉到 {len(msgs)} 条")

        # 按时间排序（有的实现给的是倒序）
        msgs.sort(key=lambda m: m["time"])

        # 规则要求：最后一条必须是别人说的；如果最后一条是机器人自己，说明已经回过
        if msgs[-1]["sender_id"] == self_id:
            note("最新一条就是机器人自己说的，已经回复过，不补")
            return ""

        # 从最新往前翻，找到机器人自己上一次发言
        bot_idx = -1
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i]["sender_id"] == self_id:
                bot_idx = i
                break
        if bot_idx < 0:
            # 翻不到机器人发言：按规则不补
            note(f"最近 {len(msgs)} 条里没有机器人（{self_id}）的发言，不补")
            return ""

        note(f"机器人上次发言在第 {bot_idx + 1}/{len(msgs)} 条")
        window = msgs[bot_idx + 1 :]
        if not window:
            note("机器人上次发言已经是最新一条，没有要补的")
            return ""
        if since_ts:
            window = [m for m in window if (m["time"] or 0) >= since_ts]
            if not window:
                note("机器人上次发言之后没有「最近」的消息（太旧了），不补")
                return ""

        async with self._seen_lock:
            seen_ids = self.seen.ids(umo)

        picked: list[str] = []
        picked_texts: list[str] = []
        picked_ids: list[str] = []
        for m in window:
            if m["id"] and m["id"] in seen_ids:
                continue
            if m["sender_id"] == self_id:
                continue
            if m["id"] and m["id"] == cur_id:
                continue
            text = m["text"].strip()
            if not text:
                continue
            # 只有拿不到 message_id 时才退化成按文本排除当前消息
            if not m["id"] and text == cur_text:
                continue
            if _already_known(text, known):
                continue
            picked.append(f"{m['name']}：{_clip(text, _PER_MSG_CHARS)}")
            picked_texts.append(text)
            if m["id"]:
                picked_ids.append(m["id"])

        if not picked:
            note(f"机器人发言后共 {len(window)} 条，去重后没有可补的")
            return ""

        if meta is not None and picked_texts:
            meta["last_text"] = picked_texts[-1]
            meta["count"] = len(picked_texts)

        # 只有真的补出去了才记「已读」，否则下次会误以为处理过
        if record and picked_ids:
            async with self._seen_lock:
                self.seen.add(umo, picked_ids)
                self.seen.save()

        max_msgs = self._cfg_int("max_messages", 30, 1)
        if len(picked) > max_msgs:
            picked = picked[-max_msgs:]
            picked_ids = picked_ids[-max_msgs:]

        body = "\n".join(picked)
        max_chars = self._cfg_int("max_chars", 3000, 200)
        # 超长时按整行从最早处丢，保证留下的是完整句子
        lines = body.split("\n")
        while len(lines) > 1 and len("\n".join(lines)) > max_chars:
            lines.pop(0)
        body = "\n".join(lines)
        if len(body) > max_chars:
            body = body[:max_chars] + "…"

        if not self.cfg.get("show_header", True):
            return body

        return (
            "【断联补读】你掉线/离开期间，有人说了下面这些话，"
            "按时间顺序（只含别人说的，你自己的发言已省略）：\n" + body
        )

    # ------------------------------------------------------ 启动即补读（主动）

    @filter.on_astrbot_loaded()
    async def on_loaded(self) -> None:
        """AstrBot 起来后自己扫一遍最近会话：断联期间漏掉的话，主动补读并回一句。"""
        if not self.cfg.get("startup_catchup", True):
            return
        logger.info(
            "[backfill] 生效配置：dry_run={} 群聊={} 私聊={} hook_hours={} startup_hours={}".format(
                bool(self.cfg.get("startup_dry_run", True)),
                bool(self.cfg.get("include_group", True)),
                bool(self.cfg.get("include_private", True)),
                self._cfg_int("hook_hours", 24, 0),
                self._cfg_int("startup_hours", 24, 1),
            )
        )
        try:
            self._boot_task = asyncio.create_task(self._startup_catchup())
        except Exception as e:
            logger.warning(f"[backfill] 启动补读没能排上：{e!r}")

    async def _startup_catchup(self) -> None:
        delay = max(5, self._cfg_int("startup_delay", 25, 5))
        auto = bool(self.cfg.get("startup_auto_delay", True))
        started = time.time()
        if auto:
            logger.info("[backfill] 启动补读：自动模式，加载完就开始探平台，探到就补")
        else:
            logger.info(f"[backfill] 启动补读：固定等待 {delay} 秒后开始探平台")
            await asyncio.sleep(delay)
        platforms: list = []
        for attempt in range(3):
            platforms = await self._wait_platforms(timeout=45)
            if platforms:
                break
            if attempt < 2:
                logger.info(
                    f"[backfill] 启动补读：第 {attempt + 1} 轮没等到平台，30 秒后再试"
                )
                await asyncio.sleep(30)
        logger.info(
            f"[backfill] 启动补读：可用平台 {len(platforms)} 个，"
            f"从加载到就绪用时 {time.time() - started:.0f} 秒"
        )
        if not platforms:
            logger.info("[backfill] 启动补读：等不到 aiocqhttp 平台，本次放弃")
            return
        for plat in platforms:
            try:
                await self._catchup_platform(plat)
            except Exception as e:
                logger.warning(f"[backfill] 启动补读失败：{e} | {traceback.format_exc()}")

    async def _wait_platforms(self, timeout: int = 120) -> list:
        """等 aiocqhttp 真的连上（能应答 get_login_info），不是"适配器对象存在"。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            out = []
            try:
                insts = self.context.platform_manager.get_insts()
            except Exception:
                insts = []
            for plat in insts or []:
                try:
                    meta = plat.meta()
                except Exception:
                    continue
                if getattr(meta, "name", "") != "aiocqhttp":
                    continue
                try:
                    client = plat.get_client()
                except Exception:
                    client = None
                if client is None:
                    continue
                try:
                    info = await asyncio.wait_for(
                        client.call_action("get_login_info"), timeout=5
                    )
                except Exception:
                    continue
                if _login_id(info):
                    out.append(plat)
            if out:
                return out
            await asyncio.sleep(3)
        return []

    async def _catchup_platform(self, plat) -> None:
        meta = plat.meta()
        client = plat.get_client()
        # 注意：CQHttp 的 .self_id 是 OneBot 动作的 partial，不是 QQ 号，别直接取
        self_id = ""
        try:
            info = await asyncio.wait_for(client.call_action("get_login_info"), timeout=10)
            self_id = _login_id(info)
        except Exception as e:
            logger.info(f"[backfill] 启动补读：get_login_info 失败（{e!r}）")
        if not self_id:
            v = getattr(client, "self_id", "")
            if isinstance(v, (str, int)):
                self_id = str(v)
        pid = str(getattr(meta, "id", "") or "")

        try:
            convs = await self.context.conversation_manager.get_conversations(
                platform_id=pid
            )
        except Exception as e:
            logger.info(f"[backfill] 启动补读：拿不到会话列表（{e!r}）")
            return
        logger.info(
            f"[backfill] 启动补读：平台 {pid} 有 {len(convs or [])} 个会话，"
            f"self_id={self_id or '(空)'}"
        )

        hours = self._cfg_int("startup_hours", 24, 1)
        now = time.time()
        since_ts = now - hours * 3600
        fresh = []
        for c in convs or []:
            umo = str(getattr(c, "user_id", "") or "")
            if not umo.startswith(f"{pid}:"):
                continue
            ts = _conv_ts(c)
            if ts and now - ts > hours * 3600:
                continue
            fresh.append((ts, umo))
        fresh.sort(reverse=True)

        limit = self._cfg_int("startup_max_sessions", 3, 1)
        dry = bool(self.cfg.get("startup_dry_run", True))
        done = 0
        for _ts, umo in fresh:
            if done >= limit:
                break
            parts = umo.split(":", 2)
            if len(parts) != 3:
                continue
            mtype, sid = parts[1], parts[2]
            group_id = sid if mtype.startswith("Group") else ""
            peer = sid if not group_id else ""
            if not group_id and not peer:
                continue
            diag: list[str] = []
            meta: dict = {}
            known = await self._history_texts(umo)
            logger.debug(f"[backfill] 启动补读扫描 {umo} ({mtype})")
            try:
                block = await self._collect(
                    bot=client,
                    self_id=self_id,
                    group_id=group_id,
                    peer=peer,
                    umo=umo,
                    known=known,
                    diag=diag,
                    record=not dry,  # 试运行不写已读，免得真开的时候反而补不到
                    since_ts=since_ts,
                    meta=meta,
                )
            except Exception as e:
                logger.info(f"[backfill] 启动补读 {umo} 出错：{e!r}")
                continue
            if not block:
                logger.info(f"[backfill] 启动补读 {umo} 无内容：{' | '.join(diag)}")
                continue
            reply = await self._ask_llm(umo, block, known)
            if not reply:
                continue
            if dry:
                logger.info(f"[backfill] 试运行：本该回复 {umo}：{reply[:80]}")
                continue
            try:
                ok = await self.context.send_message(umo, MessageChain([Plain(reply)]))
                if not ok:
                    logger.warning(f"[backfill] 启动补读：{umo} 没找到可用平台，没发出去")
                    continue
                done += 1
                logger.info(f"[backfill] 启动补读已回复 {umo}：{reply[:60]}")
                await self._save_exchange(umo, meta.get("last_text", ""), reply)
            except Exception as e:
                logger.warning(f"[backfill] 启动补读回复发送失败 {umo}：{e!r}")

    async def _save_exchange(self, umo: str, user_text: str, reply: str) -> None:
        """把这次主动回复记进会话历史：只记最后一句用户消息 + 回复。

        不记整段补读块——那是拼出来的合成文本，写进历史会一直污染后续上下文。
        """
        text = (user_text or "").strip()
        if not text:
            return
        try:
            cid = await self.context.conversation_manager.get_curr_conversation_id(umo)
            if not cid:
                logger.debug(f"[backfill] {umo} 没有会话 id，跳过写历史")
                return
            await self.context.conversation_manager.add_message_pair(
                cid,
                {"role": "user", "content": text},
                {"role": "assistant", "content": reply},
            )
            logger.debug(f"[backfill] 已把这次往返写进历史 {umo}")
        except Exception as e:
            logger.warning(f"[backfill] 写会话历史失败 {umo}：{e!r}")

    async def _history_texts(self, umo: str) -> list[str]:
        """拿会话历史里的文本，用来做去重（避免把已经聊过的再补一遍）。"""
        try:
            cid = await self.context.conversation_manager.get_curr_conversation_id(umo)
            if not cid:
                return []
            conv = await self.context.conversation_manager.get_conversation(umo, cid)
            if not conv:
                return []
            return _texts_from_messages(getattr(conv, "history", None))
        except Exception:
            return []

    async def _ask_llm(self, umo: str, block: str, known: list[str]) -> str:
        """拿人格 + 历史，让模型接着这段补读写一句回复。"""
        try:
            provider = self.context.get_using_provider(umo)
        except Exception:
            provider = None
        if provider is None:
            logger.info(f"[backfill] 启动补读：{umo} 没有可用 provider")
            return ""

        persona_prompt = ""
        try:
            persona = await self.context.persona_manager.get_default_persona_v3(umo)
            persona_prompt = getattr(persona, "system_prompt", "") or ""
        except Exception:
            pass

        contexts: list = []
        try:
            cid = await self.context.conversation_manager.get_curr_conversation_id(umo)
            if cid:
                conv = await self.context.conversation_manager.get_conversation(umo, cid)
                hist = getattr(conv, "history", None) if conv else None
                if isinstance(hist, list):
                    for m in hist[-24:]:
                        if isinstance(m, dict) and m.get("role") in ("user", "assistant"):
                            c = m.get("content")
                            if isinstance(c, str) and c.strip():
                                contexts.append({"role": m["role"], "content": c})
        except Exception:
            pass

        prompt = (
            "你刚才掉线/离开了，下面是你漏看的消息（按时间顺序）：\n"
            f"{block}\n\n"
            "接着这些消息自然地回一句，就像刚回来一样；"
            "不要提「补读」「掉线」这些词。"
        )
        kwargs = {
            "prompt": prompt,
            "contexts": contexts or None,
            "system_prompt": persona_prompt or None,
        }
        try:
            resp = await asyncio.wait_for(provider.text_chat(**kwargs), timeout=120)
        except Exception as e:
            logger.warning(f"[backfill] 启动补读 LLM 调用失败 {umo}：{e!r}")
            return ""
        return (getattr(resp, "completion_text", "") or "").strip()[:1500]

    @staticmethod
    def _inject(request, block: str) -> None:
        """把补读内容塞进本轮请求；优先用不落库的临时内容块。"""
        parts = getattr(request, "extra_user_content_parts", None)
        if TextPart is not None and isinstance(parts, list):
            try:
                parts.insert(0, TextPart(text=block).mark_as_temp())
                return
            except Exception:
                pass
        # 兜底：拼到系统提示词末尾（加分隔，别和人格提示黏在一起）
        try:
            request.system_prompt = (
                (request.system_prompt or "")
                + "\n\n===== 断联补读（临时，仅本轮有效）=====\n"
                + block
            )
        except Exception:
            pass

    async def terminate(self) -> None:
        self._last_req.clear()
        task = getattr(self, "_boot_task", None)
        if task is not None and not task.done():
            task.cancel()
        try:
            self.seen.save()
        except Exception:
            pass


# --------------------------------------------------------------------- utils


class _EmptyRequest:
    """自测命令用的假请求对象。"""

    def __init__(self) -> None:
        self.contexts: list = []
        self.extra_user_content_parts: list = []
        self.system_prompt = ""


def _login_id(payload) -> str:
    """从 get_login_info 的返回里取机器人 QQ（有的实现会再包一层 data）。"""
    data = payload
    if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
        data = payload["data"]
    if isinstance(data, dict):
        v = data.get("user_id") or data.get("uin")
        if v:
            return str(v)
    return ""


def _conv_ts(conv) -> float:
    """会话最后活跃时间（秒）。字段在不同版本里叫法不一，挨个试。"""
    for attr in ("updated_at", "created_at", "last_message_at"):
        v = getattr(conv, attr, None)
        if v is None:
            continue
        try:
            if isinstance(v, (int, float)):
                return float(v)
            return v.timestamp()
        except Exception:
            continue
    return 0.0


def _private_peer(event, self_id: str) -> str:
    """私聊时认出对方 QQ：从 umo 取，再退到 sender_id，且绝不能是自己。"""
    cands: list[str] = []
    umo = str(getattr(event, "unified_msg_origin", "") or "")
    parts = umo.split(":")
    if len(parts) >= 3:
        cands.append(parts[2].strip())
    try:
        cands.append(str(event.get_sender_id() or ""))
    except Exception:
        pass
    for c in cands:
        if c and c.isdigit() and c != self_id:
            return c
    return ""


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _normalize(payload) -> list[dict]:
    """把 OneBot 的各种返回形状统一成 [{id,sender_id,name,text,time}]。"""
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, dict):
            raw = data.get("messages") or data.get("message") or []
        elif isinstance(data, list):
            raw = data
        else:
            raw = payload.get("messages") or []
    elif isinstance(payload, list):
        raw = payload
    else:
        raw = []
    if not isinstance(raw, list):
        return []

    out: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        sender = item.get("sender") if isinstance(item.get("sender"), dict) else {}
        sender_id = str(
            sender.get("user_id")
            or item.get("user_id")
            or item.get("sender_id")
            or ""
        )
        name = str(
            sender.get("card")
            or sender.get("nickname")
            or sender_id
            or "某人"
        )
        text = _extract_text(item.get("message") or item.get("raw_message") or "")
        try:
            ts = float(item.get("time") or item.get("timestamp") or 0)
        except Exception:
            ts = 0.0
        out.append(
            {
                "id": str(item.get("message_id") or item.get("id") or ""),
                "sender_id": sender_id,
                "name": name,
                "text": text,
                "time": ts,
            }
        )
    return out


def _extract_text(message) -> str:
    """从消息段里抽纯文本；图片/表情等用占位符。"""
    if isinstance(message, str):
        return message
    if not isinstance(message, list):
        return ""
    chunks: list[str] = []
    for seg in message:
        if not isinstance(seg, dict):
            continue
        seg_type = seg.get("type")
        data = seg.get("data") if isinstance(seg.get("data"), dict) else {}
        if seg_type == "text":
            chunks.append(str(data.get("text") or ""))
        elif seg_type == "at":
            chunks.append(f"@{data.get('qq') or data.get('name') or ''}")
        elif seg_type == "image":
            chunks.append("[图片]")
        elif seg_type == "face":
            chunks.append("[表情]")
        elif seg_type == "record":
            chunks.append("[语音]")
        elif seg_type == "video":
            chunks.append("[视频]")
    return "".join(chunks)


def _context_texts(request) -> list[str]:
    """已有上下文里的用户文本，用来避免重复补读。"""
    return _texts_from_messages(getattr(request, "contexts", None))


def _texts_from_messages(items) -> list[str]:
    """从 OpenAI 格式的 messages / 会话历史里抠出文本。"""
    if isinstance(items, str):
        try:
            items = json.loads(items)
        except Exception:
            return []
    if not isinstance(items, list):
        return []
    out: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, str):
            out.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    out.append(str(part.get("text") or ""))
    # 顺手去掉「名字：」前缀，短句才好逐字比对
    for t in list(out):
        if "：" in t[:24]:
            out.append(t.split("：", 1)[1])
    return out


def _already_known(text: str, known: list[str]) -> bool:
    """这段文本是否已经在上下文里了。

    短句（<12 字）只认完全相等：像「在吗」这种到处都是的句子，
    一旦做包含匹配就会被历史里的任意一句吃掉，真正漏掉的话反而补不回来。
    """
    t = " ".join(text.split())
    if not t:
        return True
    if len(t) < 12:
        return any(t == " ".join(c.split()) for c in known)
    probe = t[:30]
    return any(probe in " ".join(c.split()) for c in known)
