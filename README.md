# stitchify

Turn photos into simplified, subject-focused cross stitch patterns.

## Why this exists

I probably need to improve my GitHub with more sophisticated and interesting projects to impress people in interviews, because in about three months I will be unemployed.

But today I am tipsy (it was a whole bottle, but let's say tipsy). I submitted my paper and crossed my fingers for an acceptance, but first, for no desk rejection.

I got a cross stitch hobby kit with a black cat, and I thought it would be cool to convert photos into minimalistic versions and then turn those into cross stitch patterns. Honestly, I didn't do an elaborate check, but the few websites I tried weren't quite what I wanted.

So, taking advantage of this bottle of wine, I decided to vibe code this project. I probably won't continue tomorrow, or won't find the time. But who cares? Here is public proof of an attempt.

Claude will probably be correcting this README because of my typos.

And if you somehow ended up in my repos and are reading this: how did you get here? Whatever you're doing right now is no better than me vibe coding a stitching pattern generator, so shoot me a message. You're my kind of people.

## Usage

```bash
pip install -r requirements.txt
python stitchify.py photo.jpg --width 80 --colors 12
```

Options: `--width` (stitches), `--colors` (max threads), `--bg-strength` (background simplification), `--no-segment` (plain conversion baseline).
