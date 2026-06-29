"""LLM API 테스트 CLI.

사용법:
  python -m scripts.llm_test                   # 대화형 REPL
  python -m scripts.llm_test "질문"            # 1회 실행
  python -m scripts.llm_test --provider gemini --model gemini-2.5-pro
  python -m scripts.llm_test --no-stream

REPL 명령:
  /system <text>   시스템 프롬프트 설정
  /reset           대화 히스토리 초기화
  /stream          스트리밍 on/off 토글
  /raw             요청 블록 표시 토글
  /json            json_mode 토글
  /quit            종료
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging


def _print_request(req: dict) -> None:
    print("\n── 요청 ──")
    print(json.dumps(req, ensure_ascii=False, indent=2))
    print("──────────")


async def _repl(provider, *, system: str | None, state: dict) -> None:
    """state keys: stream, show_raw, json_mode"""
    history: list[dict] = []
    _system = system

    print("대화를 시작합니다. /quit 또는 Ctrl-D로 종료.")
    if _system:
        print(f"[system: {_system!r}]")

    while True:
        try:
            user_input = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            parts = user_input.split(None, 1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "/quit":
                break
            elif cmd == "/reset":
                history.clear()
                print("[히스토리 초기화]")
            elif cmd == "/stream":
                state["stream"] = not state["stream"]
                print(f"[스트리밍: {'on' if state['stream'] else 'off'}]")
            elif cmd == "/raw":
                state["show_raw"] = not state["show_raw"]
                print(f"[요청 표시: {'on' if state['show_raw'] else 'off'}]")
            elif cmd == "/json":
                state["json_mode"] = not state["json_mode"]
                print(f"[json_mode: {'on' if state['json_mode'] else 'off'}]")
            elif cmd == "/system":
                _system = arg or None
                print(f"[system: {_system!r}]")
            else:
                print(f"알 수 없는 명령: {cmd}")
            continue

        history.append({"role": "user", "content": user_input})

        print("\nassistant> ", end="", flush=True)
        try:
            if state["stream"]:
                tokens: list[str] = []
                async for token in provider.stream_chat(history, system=_system):
                    print(token, end="", flush=True)
                    tokens.append(token)
                print()
                answer = "".join(tokens)
            else:
                answer = await provider.chat(
                    history, system=_system, json_mode=state["json_mode"]
                )
                print(answer)
        except Exception as e:
            print(f"\n[오류: {e}]")
            history.pop()
            continue

        history.append({"role": "assistant", "content": answer})


async def main() -> None:
    parser = argparse.ArgumentParser(description="LLM API 테스트 CLI")
    parser.add_argument("prompt", nargs="?", help="1회 실행 프롬프트 (없으면 대화형)")
    parser.add_argument("--system", default=None, help="시스템 프롬프트")
    parser.add_argument("--provider", default=None, help="provider 오버라이드 (gemini|ollama)")
    parser.add_argument("--model", default=None, help="모델 오버라이드")
    parser.add_argument("--stream", dest="stream", action="store_true", default=False)
    parser.add_argument("--no-stream", dest="stream", action="store_false")
    parser.add_argument("--no-raw", dest="show_raw", action="store_false", default=True,
                        help="요청 블록 숨기기")
    parser.add_argument("--json", dest="json_mode", action="store_true", default=False,
                        help="json_mode 활성화")
    parser.add_argument("--debug-http", action="store_true", default=False,
                        help="SDK/httpx 와이어 레벨 로깅 활성화")
    args = parser.parse_args()

    if args.debug_http:
        logging.basicConfig(level=logging.DEBUG)

    from app.config import get_settings
    from app.llm import make_llm_provider

    settings = get_settings()
    provider_name = args.provider or settings.llm_provider
    model_name = args.model or (
        settings.gemini_model if provider_name == "gemini" else settings.llm_model
    )

    # 뮤터블 state — on_request 클로저가 REPL 토글에 반응하도록
    state = {
        "stream": args.stream,
        "show_raw": args.show_raw,
        "json_mode": args.json_mode,
    }

    def on_request(req: dict) -> None:
        if state["show_raw"]:
            _print_request(req)

    provider = make_llm_provider(
        on_request=on_request,
        provider=args.provider,
        model=args.model,
    )

    print(f"[provider={provider_name}, model={model_name}, stream={state['stream']}, json_mode={state['json_mode']}]")

    if args.prompt:
        messages = [{"role": "user", "content": args.prompt}]
        print("\nassistant> ", end="", flush=True)
        if state["stream"]:
            async for token in provider.stream_chat(messages, system=args.system):
                print(token, end="", flush=True)
            print()
        else:
            answer = await provider.chat(messages, system=args.system, json_mode=state["json_mode"])
            print(answer)
    else:
        await _repl(provider, system=args.system, state=state)


if __name__ == "__main__":
    asyncio.run(main())
