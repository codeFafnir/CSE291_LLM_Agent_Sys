"""Teach the Continuum fork to pin KV for Harbor terminus-2 completions.

Terminus-2 asks the model for a JSON object with a "commands" array of
"keystrokes", not a mini-swe-agent ```bash fence. A parsed command name of
"bash" makes set_up_pin return the fork's fixed 2 second pin. Also flush
scheduler_timestamps when a request is pinned, because the stock recorder
only writes that file on shutdown.
"""

from __future__ import annotations

import sys
from pathlib import Path

PARSE_OLD = '''    def parse(self, text: str) -> Optional[str]:
        """Parse LLM output and extract the function call name.

        Args:
            text: Output text from the LLM

        Returns:
            The function call name (e.g., "ls", "cd", "git"), or None if not found
        """
        # Same regex pattern as mini-swe-agent: r"```bash\\s*\\n(.*?)\\n```"
        actions = re.findall(r"```bash\\s*\\n(.*?)\\n```", text, re.DOTALL)

        if len(actions) == 1:
            bash_action = actions[0].strip()
            # Extract the first word (command) from the action
            words = bash_action.split()
            if words:
                return words[0]

        return None
'''

PARSE_NEW = '''    def parse(self, text: str) -> Optional[str]:
        """Parse LLM output and extract the function call name.

        mini-swe-agent emits a ```bash fence. Harbor terminus-2 emits a JSON
        object whose commands carry keystrokes, sometimes tagged bash_command.
        Every bash tool shares one pin bucket named "bash".
        """
        actions = re.findall(r"```bash\\s*\\n(.*?)\\n```", text, re.DOTALL)
        if len(actions) == 1:
            words = actions[0].strip().split()
            if words:
                return words[0]
        if "bash_command" in text or "keystrokes" in text:
            return "bash"
        return None
'''

PIN_OLD = '''    def pin_request(self, request: Request, length_of_pin: float) -> None:
        self.continuum_recorder.request_pinned(request)
        self.pinned_requests.append((request, time.time() + length_of_pin))
'''

PIN_NEW = '''    def pin_request(self, request: Request, length_of_pin: float) -> None:
        self.continuum_recorder.request_pinned(request)
        self.pinned_requests.append((request, time.time() + length_of_pin))
        self.continuum_recorder.print_history()
'''


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if new in text:
        return
    if old not in text:
        raise SystemExit(f"pattern missing in {path}")
    path.write_text(text.replace(old, new, 1))


def main(clone: Path) -> None:
    estimator = clone / "vllm/v1/core/estimate_with_func.py"
    scheduler = clone / "vllm/v1/core/sched/scheduler.py"
    replace_once(estimator, PARSE_OLD, PARSE_NEW)
    replace_once(scheduler, PIN_OLD, PIN_NEW)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_continuum_harbor.py <vllm-continuum clone>")
    main(Path(sys.argv[1]))
