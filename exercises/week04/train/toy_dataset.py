import torch

#分成兩個類別，分別是0和1，每個類別有n_per_class個樣本
def make_stripe_dataset(n_per_class: int = 100,image_size: int = 32,noise_std: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    total = 2 * n_per_class #總樣本數量
    #建立total個樣本，每個樣本是一個32x32的圖像，並且有一個通道（灰度圖像），初始值為0
    images = torch.zeros(total, 1, image_size, image_size) 
    #建立標籤，初始值為空
    labels = torch.empty(total, dtype=torch.long)
    labels[:n_per_class] = 0 # 把前n_per_class個樣本的標籤設為0
    labels[n_per_class:] = 1 # 把後n_per_class個樣本的標籤設為1
    #隨機生成每個樣本的中心位置，確保線條不會超出圖像邊界
    half_width = 2
    centers = torch.randint(   
    low=half_width,
    high=image_size - half_width,
    size=(n_per_class,),
    )
    #將前n_per_class個樣本每一個圖像中間幾個 columns 設為 1，形成垂直線條
    for index, center in enumerate(centers):
        for col in range(center - half_width, center + half_width):
            images[index, :, :, col] = 1

    #將後n_per_class個樣本每一個圖像中間幾個 rows 設為 1，形成水平線條
    centers = torch.randint(
        low=half_width,
        high=image_size - half_width,
        size=(n_per_class,),
    )
    for index, center in enumerate(centers):
        for row in range(center - half_width, center + half_width):
            images[n_per_class + index, :, row, :] = 1
    #如果噪聲標準差大於0，則在圖像上添加高斯噪聲，並將圖像的值限制在0到1之間 
    if noise_std > 0:
        #生成-0.3  0.5 -1.2
        # 0.8 -0.1  0.2
        #-0.4  1.1 -0.7 類似的噪聲，其形狀與圖像相同，並且每個像素的值總和平均值為0，標準差為1*noise_std
        noise = noise_std * torch.randn_like(images)
        #將噪聲添加到圖像上，並將圖像的值限制在0到1之間
        images = torch.clamp(images + noise, 0.0, 1.0)
    return images, labels

