import datetime
import os
import json
import asyncio
import tempfile
from typing import Optional
from pathlib import Path

from telethon import TelegramClient, errors, events

# ---------------- CONFIG ----------------
API_ID = 9229596
API_HASH = "7e4a12e2e1c2bca13831c981bfa5a60c"

SOURCE_CHAT = -1003406071663
TARGET_CHAT = -1003403048131
SESSION_NAME = "mirror_session"

STATE_FILE = "mirror_state.json"
FAILED_LOG = "failed_messages.log"

BATCH_SIZE = 50
PER_MESSAGE_DELAY = 0.8
PER_BATCH_BREAK = 2.0
# ---------------------------------------

client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
HISTORY_DONE = False


def is_service_message(message) -> bool:
    return (not message) or (getattr(message, "action", None) is not None)


def load_last_id() -> int:
    if not os.path.exists(STATE_FILE):
        return 0
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return int(json.load(f).get("last_id", 0))
    except Exception:
        return 0


def save_last_id(last_id: int) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump({"last_id": int(last_id)}, f, ensure_ascii=False, indent=2)


def log_failed(msg_id: int, reason: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(FAILED_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] msg_id={msg_id} reason={reason}\n")


async def safe_send_text(text: str) -> bool:
    if not text:
        return True
    try:
        await client.send_message(TARGET_CHAT, text, parse_mode="html")
        return True
    except Exception:
        try:
            await client.send_message(TARGET_CHAT, text)
            return True
        except Exception:
            return False


def get_original_filename(message) -> str:
    """
    Best-effort original filename to help Telegram classify correctly.
    """
    name = None
    try:
        if message.file and message.file.name:
            name = message.file.name
    except Exception:
        pass

    if not name:
        # fallback name using id + inferred extension
        ext = ""
        try:
            if message.file and message.file.ext:
                ext = message.file.ext
        except Exception:
            ext = ""
        if not ext:
            ext = ".bin"
        name = f"msg_{message.id}{ext}"
    return name


async def reupload_media(message) -> bool:
    print(f"{message.id} - Downloading media from message - {datetime.datetime.now().time()}")

    # ---- download progress ----
    dl_last = -1

    def dl_progress(current, total):
        nonlocal dl_last
        if not total:
            return
        percent = int(current * 100 / total)
        if percent != dl_last:
            dl_last = percent
            print(f"   ⏬ {percent}% ({current//1024}KB / {total//1024}KB) - {datetime.datetime.now().time()}")

    # ---- upload progress ----
    ul_last = -1

    def ul_progress(current, total):
        nonlocal ul_last
        if not total:
            return
        percent = int(current * 100 / total)
        if percent != ul_last:
            ul_last = percent
            print(f"   ⏫ Upload {percent}%  - {datetime.datetime.now().time()}")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            original_name = get_original_filename(message)
            # Download using the original filename
            dest_path = str(Path(tmpdir) / original_name)

            path = await message.download_media(
                file=dest_path,
                progress_callback=dl_progress
            )

            if not path or not os.path.exists(path):
                print(f"⚠️ Download failed/empty path for message {message.id}")
                log_failed(message.id, "download_media returned no file")
                return False

            # Decide if it should be treated as video
            is_video = bool(getattr(message, "video", None))

            # ✅ Important for Desktop/Web playback:
            # - force_document=False (send as media)
            # - supports_streaming=True (for videos)
            # - file_name preserved (helps Telegram classify)
            send_kwargs = dict(
                caption=message.text or "",
                progress_callback=ul_progress,
                force_document=False,
                file_name=original_name,
            )
            if is_video:
                send_kwargs["supports_streaming"] = True

            # Send with HTML caption, fallback to plain
            try:
                await client.send_file(
                    TARGET_CHAT,
                    path,
                    parse_mode="html",
                    **send_kwargs
                )
            except Exception:
                await client.send_file(
                    TARGET_CHAT,
                    path,
                    **send_kwargs
                )

        print(f"{message.id} - Reuploaded media message - {datetime.datetime.now().time()}")
        return True

    except errors.FloodWaitError as e:
        print(f"🛑 FloodWait during media {message.id}: sleeping {e.seconds}s...")
        await asyncio.sleep(e.seconds + 1)
        return await reupload_media(message)

    except Exception as e:
        print(f"❌ Reupload failed for message {message.id}: {e}")
        log_failed(message.id, f"reupload failed: {e}")
        return False


async def download_and_send(message) -> bool:
    if is_service_message(message):
        return True

    try:
        if message.media:
            return await reupload_media(message)

        if message.text:
            ok = await safe_send_text(message.text)
            if ok:
                print(f"✅ Sent text message {message.id}")
                return True
            else:
                print(f"❌ Failed to send text message {message.id}")
                log_failed(message.id, "send_message failed")
                return False

        return True

    except errors.FloodWaitError as e:
        print(f"🛑 FloodWait on message {message.id}: sleeping {e.seconds}s...")
        await asyncio.sleep(e.seconds + 1)
        return await download_and_send(message)

    except Exception as e:
        print(f"⚠️ Error on message {message.id}: {e}")
        log_failed(message.id, f"unexpected error: {e}")
        return False


async def sync_history_from_first(resume_from_id: int):
    global HISTORY_DONE

    last_id = resume_from_id
    print(f"⏳ Syncing history from id > {last_id} (oldest → newest)")

    while True:
        msgs = await client.get_messages(
            SOURCE_CHAT,
            limit=BATCH_SIZE,
            offset_id=last_id,
            reverse=True
        )

        if not msgs:
            break

        for msg in msgs:
            if msg.id <= last_id:
                continue

            ok = await download_and_send(msg)

            # only advance when success
            if ok:
                last_id = msg.id
                save_last_id(last_id)

            await asyncio.sleep(PER_MESSAGE_DELAY)

        print(f"📦 Batch done. last_id={last_id}. Break...")
        await asyncio.sleep(PER_BATCH_BREAK)

    HISTORY_DONE = True
    print("✅ History sync complete.")


@client.on(events.NewMessage(chats=SOURCE_CHAT))
async def live_handler(event):
    if not HISTORY_DONE:
        return

    msg = event.message
    if is_service_message(msg):
        return

    ok = await download_and_send(msg)
    if ok:
        current = load_last_id()
        if msg.id > current:
            save_last_id(msg.id)

    await asyncio.sleep(PER_MESSAGE_DELAY)


async def main():
    await client.start()
    print("✅ Login successful!")

    src = await client.get_entity(SOURCE_CHAT)
    print(f"📂 Connected to Source: {getattr(src, 'title', src)}")

    env_override = "339"
    override = None  # e.g. "28"

    chosen = None
    if env_override and env_override.isdigit():
        chosen = int(env_override)
    elif override and str(override).isdigit():
        chosen = int(override)

    if chosen is not None:
        resume_from = chosen
        print(f"🟡 Override START_FROM_ID={resume_from} (starts from id>{resume_from})")
    else:
        resume_from = load_last_id()
        print(f"🧠 Resuming from saved last_id={resume_from} (starts from id>{resume_from})")

    await sync_history_from_first(resume_from)

    print("📡 Listening for new messages...")
    await client.run_until_disconnected()


if __name__ == "__main__":
    with client:
        client.loop.run_until_complete(main())