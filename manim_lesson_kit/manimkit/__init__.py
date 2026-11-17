"""Draws animkit's lesson videos with Manim.

animkit paints every lesson frame with matplotlib. This kit reads each painted frame into Manim
shapes (``recorder``), follows every shape from frame to frame (``timeline``), and plays the
changes as Manim animations (``motion``, ``player``): each chapter ends on exactly the picture
the matplotlib video shows, and Manim moves the shapes in between. ``director`` plays a whole
lesson, chapter by chapter, for any project's ``Scene``.

The modules, in the order the data flows through them:

- ``frame``: where a figure's pixels land on a Manim frame.
- ``paths``, ``glyphs``: matplotlib paths and text as Manim's cubic Bézier points.
- ``leaves``, ``batches``, ``recorder``: one drawn figure as a list of leaves.
- ``timeline``: leaves followed across frames as lives.
- ``motion``: lives planned as tracks (appear, morph, cut, fade out).
- ``camera``, ``player``: tracks played on a Manim scene, frame by frame.
- ``director``: a whole lesson, into an MP4.
- ``cli``: animkit's lesson command line, with Manim as the player.
- ``parity``: the checks that the Manim lesson ends every chapter on matplotlib's picture.
"""
