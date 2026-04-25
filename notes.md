- `scp nadine.hage-chehade@cher.polytechnique.fr:~/MODAL/Challenge/DeepLearningModal/submission.csv C:\EcolePolytechnique\2A\P3\Modal\Challenge\`

- `nvidia-smi` : to check GPU usage

# Tmux
- `tmux new -s session_name`
- ctrl+b puis d pour sortir de la session
- `tmux attach -t session_name` pour revenir


# AI Prompt
- Dataset information : something to something action detection
    - training set : about 50000 videos
    - each video is made up of 4 frames
    - each frame is a 224 x 224 colored image
- Desired architecture : 
    - 2D convolution
    - represent each image by 16 tokens of flattened patches from the image
    - multi-head masked attention on these tokens