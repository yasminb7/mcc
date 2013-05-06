#!/bin/bash

mencoder 'mf://frame_*.png' -mf type=png:fps=25 -ovc lavc -lavcopts vcodec=mpeg4 -oac copy -o animation.mpg

