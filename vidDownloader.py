import linkParser
import flujo
import os

LinkParser = linkParser.videoFile("./videos.txt")

print(LinkParser.getVideoLinks())
