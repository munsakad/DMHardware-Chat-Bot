"""Traceable lesson passages and explicitly generated engineering exercises.

No paraphrase augmentation or repeated rows. The snapshot is the owner's
Hardware Education Hub catalog at commit c5fa04d (120 lessons).
"""
import hashlib
import json
import math
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CATEGORIES = {
    "electricity-electronics-basics": "Electronics", "arduino": "Electronics",
    "digital-logic-design": "Digital Logic Design", "verilog": "Digital Logic Design",
    "vhdl": "Digital Logic Design", "pcb-design": "PCB Design", "robotics": "Robotics",
}


class LessonBlocks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = None
        self.buffer = []
        self.heading = "Introduction"
        self.blocks = []

    def handle_starttag(self, tag, attrs):
        if self.active is None and tag in ("h2", "h3", "h4", "p", "li", "pre", "table"):
            self.active = tag
            self.buffer = []
        elif self.active and tag in ("br", "tr", "td", "th"):
            self.buffer.append("\n" if tag in ("br", "tr") else " | ")

    def handle_data(self, text):
        if self.active:
            self.buffer.append(text)

    def handle_endtag(self, tag):
        if tag != self.active:
            return
        raw = "".join(self.buffer).strip()
        text = raw if tag == "pre" else " ".join(raw.split())
        if tag in ("h2", "h3", "h4"):
            self.heading = text
        elif 80 <= len(text) <= 3000:
            self.blocks.append((self.heading, text, "code" if tag == "pre" else "lesson_passage"))
        self.active = None


def lesson_records():
    courses = json.loads((ROOT / "data/source_lessons.json").read_text(encoding="utf-8"))
    seen = set()
    for course in courses:
        for lesson in course["lessons"]:
            parser = LessonBlocks()
            parser.feed(lesson["content"])
            for heading, content, kind in parser.blocks:
                normalized = " ".join(content.split()).lower()
                if normalized in seen:
                    continue
                seen.add(normalized)
                digest = hashlib.sha256(content.encode()).hexdigest()[:8]
                url = ("https://munsakad.github.io/Hardware-Engineering-Education-Hub/pages/course.html"
                       f"?slug={course['slug']}&lesson={lesson['slug']}")
                yield {"category": CATEGORIES[course["slug"]], "subcategory": course["title"],
                       "title": f"{lesson['title']} — {heading} [{digest}]", "content": content,
                       "tags": f"{course['slug']};{lesson['slug']};{kind}", "source_url": url,
                       "source_group": f"lesson:{course['slug']}/{lesson['slug']}", "record_type": kind}


def worked_examples():
    """128 distinct numerical exercises; not measured hardware observations."""
    for n in range(16):
        supply = [3.3, 5, 9, 12][n % 4]
        current = [0.002, 0.005, 0.008, 0.01][n // 4]
        resistance = (supply - 2) / current
        yield example("Electronics", "LED current limiting", n,
                      f"For an ideal {supply:g} V supply, an assumed LED forward drop of 2 V and a target current of {current*1000:g} mA, "
                      f"R=(Vs−Vf)/I=({supply:g}−2)/{current:g}={resistance:g} ohms. Resistor dissipation is I²R={current**2*resistance:.4g} W. "
                      "Choose the next suitable standard resistance, then recheck current and power at supply and LED tolerances; verify the actual LED datasheet.")
        r1, r2 = [1000, 2200, 4700, 10000][n % 4], [1000, 3300, 6800, 22000][n // 4]
        out = 5*r2/(r1+r2)
        yield example("Electronics", "Unloaded voltage divider", n,
                      f"With a 5 V input, upper resistor R1={r1} ohms and lower resistor R2={r2} ohms, the unloaded output is "
                      f"Vout=Vin×R2/(R1+R2)={out:.4g} V. The Thevenin output resistance is R1||R2={r1*r2/(r1+r2):.4g} ohms. "
                      "This assumes ideal resistors and no load; a finite load parallels R2 and reduces the output.")
        r, c = [1000, 3300, 10000, 47000][n % 4], [1e-9, 10e-9, 100e-9, 1e-6][n // 4]
        yield example("Electronics", "RC low-pass cutoff", n,
                      f"For an ideal first-order RC low-pass with R={r} ohms and C={c*1e9:g} nF, "
                      f"fc=1/(2πRC)={1/(2*math.pi*r*c):.5g} Hz and the time constant RC={r*c:.5g} s. "
                      "The amplitude at fc is 1/√2 of the low-frequency amplitude (about −3.01 dB). Source resistance and load impedance alter the real response.")
        yield example("Electronics", "Capacitor step settling", n,
                      f"An initially discharged ideal capacitor of {c*1e9:g} nF charges through {r} ohms from a 3.3 V step. "
                      f"The time constant is τ=RC={r*c:.5g} s. Reaching 99% of the final voltage takes t=−RC ln(0.01)={-r*c*math.log(.01):.5g} s. "
                      "This assumes a first-order circuit with no additional load or current limit.")
        rp, bus = [1000, 2200, 4700, 10000][n % 4], [50, 100, 200, 400][n // 4]
        yield example("PCB Design", "Open-drain bus rise time", n,
                      f"For an open-drain bus with pull-up resistance {rp} ohms and estimated capacitance {bus} pF, the first-order "
                      f"30%–70% rise time is tr=ln(0.7/0.3)RC≈0.8473RC={0.84729786*rp*bus*1e-3:.4g} ns. "
                      f"At 3.3 V, ideal low-state pull-up current is about {3300/rp:.4g} mA. Check the protocol timing and the device's sink-current and VOL limits; this estimate does not certify compliance.")
        bits, reference = [8, 10, 12, 16][n % 4], [1.1, 2.048, 3.3, 5][n // 4]
        yield example("Electronics", "Ideal ADC quantization", n,
                      f"An ideal {bits}-bit ADC with a {reference:g} V reference has 2^{bits}={2**bits} quantization bins. "
                      f"Using the ideal bin-width convention, one LSB=Vref/2^N={reference/2**bits:.6g} V. "
                      "This is quantization spacing, not guaranteed accuracy. Reference error, noise, input acquisition, offset and gain error can dominate; consult the converter's transfer-function convention.")
        pulses, edges = [100, 200, 500, 1000][n % 4], [50, 100, 250, 500][n // 4]
        yield example("Robotics", "Encoder speed estimate", n,
                      f"An encoder produces {pulses} counted edges per revolution at the chosen decoding mode. If {edges} edges are measured in 0.1 s, "
                      f"speed=edges/(edges-per-revolution×interval)={edges/(pulses*.1):.5g} rev/s, or {60*edges/(pulses*.1):.5g} RPM. "
                      "Use counts per revolution after quadrature decoding, not a vendor's ambiguous PPR label, and account for any gearbox ratio.")
        prescaler, top = [1, 8, 64, 256][n % 4], [99, 249, 999, 1999][n // 4]
        yield example("Digital Logic Design", "Counter event frequency", n,
                      f"A 16 MHz timer clock divided by prescaler {prescaler} drives a counter that emits an event after counting from 0 through {top} inclusive. "
                      f"There are TOP+1={top+1} ticks per event, so fevent=16000000/({prescaler}×{top+1})={16000000/(prescaler*(top+1)):.6g} Hz. "
                      "A pin toggled once per event has half this square-wave frequency. Verify the timer mode, register width and clock source for actual hardware.")


def example(category, topic, number, content):
    return {"category": category, "subcategory": "Worked calculations", "title": f"{topic}: case {number+1:02d}",
            "content": content, "tags": f"worked_example;{topic.lower().replace(' ', '_')}",
            "source_url": "https://github.com/munsakad/DMHardware-Chat-Bot/blob/main/expand_knowledge.py",
            "source_group": "calculation:" + topic, "record_type": "generated_worked_example"}


def expanded_records():
    return [*lesson_records(), *worked_examples()]
