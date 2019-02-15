# python3 main.py train ae_01 -epoch_num 100 -save ae_01.pkl -log ae_01.txt
# python3 main.py predict ae_01 -load ../weights/ae_01.pkl -predict_dir prediction/ae_01/
# python3 main.py train ae_02 -epoch_num 100 -save ae_02.pkl -log ae_02.txt
# python3 main.py train vae_01 -epoch_num 500 -save vae_01_01.pkl -log vae_01_01.txt -sigma 0.5
# python3 main.py predict vae_01 -load ../weights/vae_01_01.pkl -predict_dir prediction/vae_01_01/
# python3 main.py train vae_01 -epoch_num 500 -save vae_01_02.pkl -log vae_01_02.txt -sigma 0.05
# python3 main.py predict vae_01 -load ../weights/vae_01_02.pkl -predict_dir prediction/vae_01_02/
# python3 main.py train vae_01 -epoch_num 150 -save vae_01_03.pkl -log vae_01_03.txt -sigma 0.001
# python3 main.py predict vae_01 -load ../weights/vae_01_03.pkl -predict_dir prediction/vae_01_03/
# python3 main.py train vae_01 -epoch_num 150 -save vae_01_04.pkl -log vae_01_04.txt -sigma 0.0001
# python3 main.py predict vae_01 -load ../weights/vae_01_04.pkl -predict_dir prediction/vae_01_04/
# python3 main.py train vae_02 -epoch_num 150 -save vae_02_01.pkl -log vae_02_01.txt -sigma 0.001
# python3 main.py predict vae_02 -load ../weights/vae_02_01.pkl -predict_dir prediction/vae_02_01/
# python3 main.py train vae_02 -epoch_num 150 -save vae_02_02.pkl -log vae_02_02.txt -sigma 0.0001
# python3 main.py predict vae_02 -load ../weights/vae_02_02.pkl -predict_dir prediction/vae_02_02/
# python3 main.py train vae_01 -epoch_num 400 -save vae_01_05.pkl -log vae_01_05.txt -sigma 0.0001
python3 main.py predict vae_01 -load ../weights/vae_01_05.pkl -predict_dir prediction/vae_01_05/
# python3 main.py train vae_02 -epoch_num 400 -save vae_02_03.pkl -log vae_02_03.txt -sigma 0.0001
python3 main.py predict vae_02 -load ../weights/vae_02_03.pkl -predict_dir prediction/vae_02_03/