python train_model.py --model_name RSS_soft --learning_rate 1e-4 --rss_factor 1.0 --soft_routing_argmax soft
python train_model.py --model_name RSS_OT_RP_soft --learning_rate 1e-2 --rss_factor 1.0 --ot_factor 1 --rp_factor 1000000 --soft_routing_argmax soft
python train_model.py --model_name BT_soft --learning_rate 1e-2 --bt_factor 1.0 --soft_routing_argmax soft
