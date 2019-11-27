# sample to run LSTM_LM

## extract
```
python LSTM_convert.py --source weathernews.txt --format word --repeat 2
python LSTM_convert.py --source weathernews_state.txt --format state --repeat 2 --output weathernews.npz
```

## training
```
python LSTMLM_train.py --model model.h5 --dataset weathernews.npz --batch_size 128 --latent_node 128 --epoch 1000
python LSTMLM_train.py --model model.h5 --dataset weathernews.npz --batch_size 128 --latent_node 128 --epoch 1000 --continue_train
```

## generating
```
python LSTMLM_generate.py --model ./model.h5 --dataset weathernews.npz --format word --size 10
python LSTMLM_generate.py --model ./model.h5 --dataset weathernews.npz --format word --size 10 --without_BOS --without_EOS
python LSTMLM_generate.py --model ./model.h5 --dataset weathernews.npz --format word --size 10 --separator ,
python LSTMLM_generate.py --model ./model.h5 --dataset weathernews.npz --format state --size 10 --offset -1 --without_BOS --without_EOS
```
