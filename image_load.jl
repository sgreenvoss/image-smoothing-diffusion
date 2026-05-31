using TestImages, Images
using ImageView


function rand_img(x, y)
    return rand(x,y)
end

function save_img(name, image)
    save("$name.png")
end
