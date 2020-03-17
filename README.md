# surat

Implementation based on this publication:

https://research.nvidia.com/publication/2017-07_Audio-Driven-Facial-Animation


Modified, to try with morph tagets as output. However, that data wasn't enough so my latest commits are for per vertex data.
Also modified, so it uses MFCC instead of auto-correlations.


https://vimeo.com/338394571

Model for this video is trained with about 30 seconds of data, which is 1/10th of what was used for the publication above.
Clips in the video are for validation. They don't appear in training data.


Rig is used from this IOS app:

http://www.bannaflak.com/face-cap/
