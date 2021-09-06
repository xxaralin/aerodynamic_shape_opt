# Nil-aerodynamic shape optimization (TD3 ve SAC)
# TD3
Kod deneme_td3.py adlı dosyada.\
Dosya terminalde şu şekilde çalıştırılır
```
python deneme_td3.py
```
Kodda denediğim hiperparametreler ceza katsayısı olan bıdık, noise, tau, learning rate oldu.\
CEZA KATSAYISI (BIDIK)- 0 ve 1 arasında değerler ve 1 ile 100 arasında aralıklı değerler denedim
![TD3-bıdık](https://user-images.githubusercontent.com/67866767/132174488-adfd79c7-d120-4717-9fd3-56aaa9bba15e.png)

Learning rate- 1e-6 dan 3e-3 arasında değerler denedim
![TD3-lr](https://user-images.githubusercontent.com/67866767/132174586-bfa4f1d5-daab-44f7-bbfd-acc5f754dc2c.png)


Noise - 1e-6 dan 5e-1 arasında değerler denedim
![TD3-noise](https://user-images.githubusercontent.com/67866767/132174711-0a6c339c-be61-4ed4-901b-0e5020871674.png)

tau-1e-3 ve 1e-2 arası değerler denedim
![TD3-tau](https://user-images.githubusercontent.com/67866767/132174766-afa5c1d2-a034-44ea-aaab-6606b3e23541.png)



# SAC
SAC algoritmasına ait dosyalar nil2 adlı klasörde. \
sac2q.py dosyası koda ait.
Kodu çalıştırmak için terminale
```
python3.7 sac2q.py
```
Kodun içinde kullanılan hard_update ve soft_update dosyası utils dosyasında tanımlı.\
Kodda denediğim hiperparametreler ceza katsayısı, alpha, deterministic vs Gaussian, lr, katman ve nöronlar

CEZA KATSAYISI(BIDIK)- 0 ve 1 arasında değerler ve 1 ile 100 arasında aralıklı değerler denedim
![SAC-bıdık](https://user-images.githubusercontent.com/67866767/132174969-2f6fa97d-1fac-4ec8-bd3b-405e8ee64f1a.png)

Learning rate- 1e-6 dan 3e-3 arasında değerler denedim
![SAC-lr](https://user-images.githubusercontent.com/67866767/132175058-7cf07ff3-48a7-4122-8ef4-a4a1208a935e.png)

alpha-0.01 ve 0.4 arasında değerler denedim
![SAC-alpha](https://user-images.githubusercontent.com/67866767/132175115-59fd4c99-dce2-4179-bf05-8372443df478.png)

deterministic ve gaussian
![SAC-noise](https://user-images.githubusercontent.com/67866767/132175223-8488eaf4-3373-4d58-9d1b-390371400a8e.png)
 
 orijinalde (8,200,200) nöronlara sahip olan 3 katmanlı networkü (8,200,100,50) nöronlu 4 katmanlı networke çevirdim
 ![SAC-layer](https://user-images.githubusercontent.com/67866767/132175313-57d50867-468c-4476-a597-5e7777d0ab5c.png)
)



# Kaynaklar
https://github.com/djbyrne/TD3/blob/master/TD3.ipynb

https://github.com/DLR-RM/rl-baselines3-zoo
