"""The official TypeSafe SDK, unchanged, answered by a local model through `s1ght serve`.

pip install -e ".[serve]" typesafe-sdk
s1ght serve model=d1-omni-600m port=11436          # in another terminal
python examples/typesafe_client.py bus.jpg
"""

import os
import sys
from pathlib import Path

from typesafe_sdk import Choice, Image, Noul, TypeSafeClient

os.environ.setdefault("TYPESAFE_BASE_URL", "http://127.0.0.1:11436")
# the SDK needs a key; s1ght serve ignores it unless started with api_key=
os.environ.setdefault("TYPESAFE_API_KEY", "local")

photo = Image(Path(sys.argv[1] if len(sys.argv) > 1 else "bus.jpg"))  # the SDK takes a Path or bytes, not a str
with TypeSafeClient(timeout=300) as client:
    print("served:", [m.name for m in client.models.list().models])
    r = client.system_one(
        model="jev-latest",  # any name: s1ght serve answers with the model it was started with
        state={"photo": photo},
        questions={
            "bus": Noul(instructions="Is there a bus in `photo`?"),
            "scene": Choice(
                instructions="Where was `photo` taken?",
                criteria={"city street": None, "football pitch": None, "kitchen": None},
            ),
        },
    )
print(f"{r.model}: bus={r.nouls['bus'].noul:.3f} scene={r.choices['scene'].choice} "
      f"({r.choices['scene'].confidence:.2f})")  # fmt: skip
