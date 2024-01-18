import os

datasets = ['parler', 'echo', 'gab']
processed_data_output_dir = {dataset: os.path.join(f'Dataset/{dataset.capitalize()}Data/') for dataset in datasets}

raw_data_path_config = {
    "parler": {
        "replies": "/sise/Yalla_work/data/parler/comments_edge_dict.p",
        "reposts": "/sise/Yalla_work/data/parler/echos_edge_dict.p",
        "users_labels": "/sise/home/tommarz/parler-hate-speech/emnlp23/parler_users_2_labels_for_gnn.tsv",
        "users_posts": "/sise/Yalla_work/data/parler/concise_dfs",
        # "users_info": "/sise/Yalla_work/data/parler/parler_users"
        "users_info": "/sise/Yalla_work/data/parler/users_info_as_df.p"
    },
    "echo": {
        "replies": "/sise/Yalla_work/data/echoes/only_english/dfs_and_dicts/el_echo_users_replies.txt",
        "reposts": "/sise/Yalla_work/data/echoes/only_english/dfs_and_dicts/el_echo_users_rt.txt",
        "mentions": "/sise/Yalla_work/data/echoes/only_english/dfs_and_dicts/el_echo_users.txt",
        "users_labels": "/sise/home/tommarz/hate_speech_detection/data/user_level/echo_users_2_labels.tsv",
        # "users_posts": "/sise/Yalla_work/data/echoes/only_english/recent_history",
        "users_posts_list": "/sise/home/tommarz/hate_speech_detection/hate_networks/outputs/echo_networks/pickled_data/corpora_list_per_user.pkl",
        "users_info": "/sise/Yalla_work/data/echoes/only_english/dfs_and_dicts/echo_usr_nodes_df.tsv"
    },
    "gab": {
        "replies": "/sise/home/tommarz/hate_speech_detection/hate_networks/outputs/gab_networks/network_data/edges/comment_edges_df.tsv",
        "reposts": "/sise/home/tommarz/hate_speech_detection/hate_networks/outputs/gab_networks/network_data/edges/retweet_edges_df.tsv",
        "mentions": "/sise/home/tommarz/hate_speech_detection/hate_networks/outputs/gab_networks/network_data/edges/mention_edges_df.tsv",
        "users_labels": "/sise/home/tommarz/hate_speech_detection/data/user_level/gab_users_2_labels.tsv",
        # "users_posts": "/sise/Yalla_work/data/gab/only_english/recent_history",
        "users_posts_list": "/sise/home/tommarz/hate_speech_detection/hate_networks/outputs/gab_networks/pickled_data/corpora_list_per_user.pkl",
        # "users_info": "/sise/Yalla_work/data/gab/only_english/dfs_and_dicts/gab_usr_nodes_df.tsv"
    }
}

processed_data_path_config = {
    'parler': {
        'network': 'Dataset/ParlerData/network.weighted.edgelist.gz',
        'network_nodes': 'Dataset/ParlerData/network_nodes.p',
        'ego_network': 'Dataset/ParlerData/ego_network.weighted.edgelist.gz',
        'ego_network_nodes': 'Dataset/ParlerData/ego_network_nodes.p',
        'users_info': 'Dataset/ParlerData/users_info.p',
        'users_posts': "/sise/Yalla_work/data/parler/gnn/users_posts_df.p",
        'users_posts_list': "/sise/Yalla_work/data/parler/gnn/users_posts_list.p",
        # 'users_posts_dict_proc': 'Dataset/ParlerData/users_posts_dict_proc.p',
        'users_with_posts': 'Dataset/ParlerData/users_with_posts.p',
        'user_sentences_proc': '/sise/Yalla_work/data/parler/gnn/users_sentences_proc.p',
        'replies': 'Dataset/ParlerData/replies.tsv',
        'reposts': 'Dataset/ParlerData/reposts.tsv',
        'mentions': 'Dataset/ParlerData/mentions.tsv',
        'edges': 'Dataset/ParlerData/edges.tsv',
        'doc2vec': 'Dataset/ParlerData/doc2vec.p',
        'dataset': 'Dataset/ParlerData/dataset.pt'
    },
    'echo': {
        'network': 'Dataset/EchoData/network.weighted.edgelist.gz',
        'network_nodes': 'Dataset/EchoData/network_nodes.p',
        'ego_network': 'Dataset/EchoData/ego_network.weighted.edgelist.gz',
        'ego_network_nodes': 'Dataset/EchoData/ego_network_nodes.p',
        'users_info': 'Dataset/EchoData/users_info.p',
        'users_posts': 'Dataset/EchoData/users_posts.p',
        'users_posts_df': 'Dataset/EchoData/users_posts_df.p',
        'users_posts_list': 'Dataset/EchoData/users_posts_list.p',
        'users_posts_dict_proc': 'Dataset/EchoData/users_posts_dict_proc.p',
        'users_with_posts': 'Dataset/EchoData/users_with_posts.p',
        'user_sentences_proc': 'Dataset/EchoData/user_sentences_proc.p',
        'replies': 'Dataset/EchoData/replies.tsv',
        'reposts': 'Dataset/EchoData/reposts.tsv',
        'mentions': 'Dataset/EchoData/mentions.tsv',
        'edges': 'Dataset/EchoData/edges.tsv',
        'doc2vec': 'Dataset/EchoData/doc2vec.p',
        'dataset': 'Dataset/EchoData/dataset.pt'
    },
    'gab': {
        'network': 'Dataset/GabData/network.weighted.edgelist.gz',
        'network_nodes': 'Dataset/GabData/network_nodes.p',
        'ego_network': 'Dataset/GabData/ego_network.weighted.edgelist.gz',
        'ego_network_nodes': 'Dataset/GabData/ego_network_nodes.p',
        'users_info': 'Dataset/GabData/users_info.p',
        'users_posts': 'Dataset/GabData/users_posts.p',
        'users_posts_df': 'Dataset/GabData/users_posts_df.p',
        'users_posts_list': 'Dataset/GabData/users_posts_list.p',
        'users_posts_dict_proc': 'Dataset/GabData/users_posts_dict_proc.p',
        'users_with_posts': 'Dataset/GabData/users_with_posts.p',
        'user_sentences_proc': 'Dataset/GabData/user_sentences_proc.p',
        'replies': 'Dataset/GabData/replies.tsv',
        'reposts': 'Dataset/GabData/reposts.tsv',
        'mentions': 'Dataset/GabData/mentions.tsv',
        'edges': 'Dataset/GabData/edges.tsv',
        'doc2vec': 'Dataset/GabData/doc2vec.p',
        'dataset': 'Dataset/GabData/dataset.pt'
    }
}
