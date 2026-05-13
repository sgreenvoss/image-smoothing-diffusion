using TestImages, Images
using ImageView

# gives grayscale image as 2d arr
x, y = 200, 200
img = rand(x, y)
img2 = copy(img)

for i in 1:x
    for j in 1:y
        img2[i,j] *= 0.5
    end
end
comb = [img img2]
save("both_images.png", comb)

#print("testing, hello world!\n")