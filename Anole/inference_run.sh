# # text -> image
# python text2image.py \
#     -i 'draw a dog' \
#     -s '/lid/home/saydalie/multimodal_cot/anole/outputs/' \
#     -b 1

# # text -> text + image
# python interleaved_generation.py \
#     -i 'Show me how to make salad with images.' \
#     -s '/lid/home/saydalie/multimodal_cot/anole/outputs/'

# # text + image -> text + image
# python inference.py \
#     -i input.json \
#     -s '/lid/home/saydalie/multimodal_cot/anole/outputs/'
