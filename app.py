import random
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    eta = None
    traffic_level = None
    explanation = None
    bus_line = ""
    stop = ""

    if request.method == "POST":
        bus_line = request.form.get("bus_line", "").strip()
        stop = request.form.get("stop", "").strip()

        # Pick traffic level first, then generate ETA from its range.
        traffic_options = {
            "low": (3, 6),
            "medium": (6, 10),
            "high": (10, 15),
        }
        traffic_level = random.choice(list(traffic_options.keys()))
        min_eta, max_eta = traffic_options[traffic_level]
        eta = random.randint(min_eta, max_eta)
        explanation = f"Traffic is {traffic_level}, expected arrival updated."

    return render_template(
        "index.html",
        eta=eta,
        traffic_level=traffic_level,
        explanation=explanation,
        bus_line=bus_line,
        stop=stop,
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
