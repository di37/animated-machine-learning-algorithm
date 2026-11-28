"""Series videos: a playlist's introduction and conclusion, made of full-screen cards.

A lesson shows one algorithm learning; a series video frames the nine lessons of a playlist: who the
series is for, the data, the algorithms and how each lesson runs (the introduction), then how they all
did and what that teaches (the conclusion). They use the lessons' stage, theme, cards and voice, so they
look and sound like the lessons around them.

- ``chapters``: a series as a list of chapters, each a length, a painter and the words it says.
- ``cards``: the text cards (a headline with lines, the algorithm line-up, the chapter road map, the
  colour key).
- ``charts``: the cards that draw data (a scatter of the real rows, a scoreboard, before-and-after rows).
- ``render``: plays a series into an MP4 with matplotlib, and writes the chapters file the voice reads.
- ``voice``: speaks each chapter's words and dubs them over the video, as the lessons' voice-over does.

The words and numbers belong to each dataset's project; this package names no dataset.
"""
