# для импорта из родительского каталога
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import pandas as pd
from load_model2 import MetricsModel
from plot_efficiency import plot_EfficiencyDouble
from richgan.schemas import _create_config_and_update_from_file
from math import sqrt


# вычисляем расстояние по идее Артема
def calculate_dist(bins):
    dis_sum = 0
    for bin_x in bins.iterrows():
        eff_ratio = bin_x[1][0]
        err_low = bin_x[1][1]
        err_high = bin_x[1][2]
        if eff_ratio <= 1:
            dist = ((1-eff_ratio)/err_high)**2
        else:
            dist = ((1-eff_ratio)/err_low)**2
        dis_sum += dist
    return dis_sum

# если брать корень в самом конце
def dis_for_model(raw_output_dict, model_name, percents):
    dis_sum = 0
    for eff_name in list(raw_output_dict['EfficiencyMaker:1'][model_name]):
        # чтобы не было слишком сильной корреляции
        if '_reverse' in eff_name or '_P_T_' in eff_name:
            continue
        all_percent_dist = 0
        for percent in percents:
            eff_ratio_column = 'eff_ratio_' + str(percent)
            eff_ratio_err_low_column = 'eff_ratio_err_low_' + str(percent)
            eff_ratio_err_high_column = 'eff_ratio_err_high_' + str(percent)
            bins = raw_output_dict['EfficiencyMaker:1'][model_name][eff_name][[eff_ratio_column,
                                                                               eff_ratio_err_low_column,
                                                                               eff_ratio_err_high_column]]
            percent_dist = calculate_dist(bins)
            # складываем для всех процентов
            all_percent_dist += percent_dist

        dis_sum += all_percent_dist
    return sqrt(dis_sum)


# Если брать корень после каждого графика
def dis_for_model_end(raw_output_dict, model_name, percents):
    dis_sum = 0
    for eff_name in list(raw_output_dict['EfficiencyMaker:1'][model_name]):
        # чтобы не было слишком сильной корреляции
        if '_reverse' in eff_name or '_P_T_' in eff_name:
            continue
        all_percent_dist = 0
        for percent in percents:
            eff_ratio_column = 'eff_ratio_'+str(percent)
            eff_ratio_err_low_column = 'eff_ratio_err_low_'+str(percent)
            eff_ratio_err_high_column = 'eff_ratio_err_high_'+str(percent)
            bins = raw_output_dict['EfficiencyMaker:1'][model_name][eff_name][[eff_ratio_column,
                                                                               eff_ratio_err_low_column,
                                                                               eff_ratio_err_high_column]]
            percent_dist = calculate_dist(bins)
            # складываем для всех процентов
            all_percent_dist += percent_dist

        dis_sum += sqrt(all_percent_dist)
    return dis_sum


def compare_metrics(efficiency_dict):
    percents_list = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    dist_list = {'GAN': [], 'KDE': []}
    for percent in percents_list:
        dis_sum = dis_for_model(raw_output_dict = efficiency_dict, model_name='GAN', percents=[percent])
        dist_list['GAN'].append(dis_sum)
        dis_sum = dis_for_model(raw_output_dict = efficiency_dict, model_name='KDE', percents=[percent])
        dist_list['KDE'].append(dis_sum)


    dist_list2 = {'GAN': [], 'KDE': []}
    for percent in percents_list:
        dis_sum = dis_for_model_end(raw_output_dict = efficiency_dict, model_name='GAN', percents=[percent])
        dist_list2['GAN'].append(dis_sum)
        dis_sum = dis_for_model_end(raw_output_dict = efficiency_dict, model_name='KDE', percents=[percent])
        dist_list2['KDE'].append(dis_sum)


    return  dist_list, dist_list2

def get_metrics(config):
    m_model = MetricsModel(config)
    targets_data_dict = m_model.get_targets()
    # нужно взять все эффетивности
    thresholds = [[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
                  [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
                  [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
                  [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]]

    efficiency_dict = plot_EfficiencyDouble(targets_data_dict['features'],
                                            targets_data_dict['targets_real'],
                                            targets_data_dict['targets_fake_gan'],
                                            targets_data_dict['targets_fake_kde'],
                                            targets_data_dict['weights'],
                                            thresholds=thresholds,
                                            model_names=['GAN  (no wght)', 'KDE'])

    dist_list, dist_list2 = compare_metrics(efficiency_dict)

    return dist_list, dist_list2

if __name__ == "__main__":

    from pathlib import Path
    from os import listdir

    # base_dir = r'C:\Users\Sergey\Sources\temp\kde_experiment'
    # datest_path = r'C:\Users\Sergey\Sources\temp\synthetic_dataset'
    base_dir = r'/home/smohnenko/RICH_GAN/kde_experiment'
    datest_path = r'/home/smohnenko/RICH_GAN/data/synthetic_dataset'


    dir_prefix = 'saved_models_aug'
    # original_config_file = r'..\richgan\configs\simple.mc\gan_vs_kde.config.yaml'
    original_config_file = r'/home/smohnenko/RICH_GAN/rich-gan-tf-2021/richgan/configs/simple.mc/gan_vs_kde.config.yaml'
    # chpt_dir = r'..\saved_models'

    kwargs = {}
    original_config = _create_config_and_update_from_file(original_config_file, **kwargs)

    all_gan_list = []
    all_kde_list = []

    full_list = []


    for num in range(0, 10):
        base_path = os.path.join(base_dir, dir_prefix+str(num))
        save_base_path = os.path.join(base_path,listdir(base_path)[0])

        # проверка папки на пустую
        if len(os.listdir(save_base_path)) <= 2:
            print('Пустая папка:')
            print(save_base_path)
            # забиваем нулями
            all_gan_list.append(0)
            all_kde_list.append(0)
            full_list.append(0)
            continue
        else:
            print('Загружаем папку:')
            print(save_base_path)

        config_file = os.path.join(save_base_path, 'config.yaml')
        log_path = Path(os.path.join(base_dir, dir_prefix+str(num)+'_log'))

        config = _create_config_and_update_from_file(config_file, **kwargs)
        # мержим два конфига


        original_config['create_data_manager']['augmentations'] = config['create_data_manager']['augmentations']
        original_config['create_data_manager']['data_path'] = datest_path
        original_config['create_data_manager']['extra_sample_config']['particle'] = 'test'
        original_config['create_data_manager']['extra_sample_config']['path'] = datest_path

        # не написано про загрузку чекпоината
        # Loading cp: ../saved_models/SimpleModelMuonMC/epoch-004999
        # должно быть перед Test dataset size: 117804
        # может эпохи нужны
        original_config['create_gan']['name'] = listdir(base_path)[0]

        original_config['create_training_manager']['save_base_path'] = base_path
        original_config['create_training_manager']['log_path'] = log_path

        # original_config['create_training_manager']['epochs'] = 833
        # original_config['create_training_manager']['save_interval_in_epochs'] = \
        #     config['create_training_manager']['save_interval_in_epochs']


        dist_list, dist_list2 = get_metrics(original_config)
    
        print('Берем корень в самом конце:')
        print('GAN:', dist_list['GAN'][-1])
        print('KDE:', dist_list['KDE'][-1])

        all_gan_list.append(dist_list['GAN'][-1])
        all_kde_list.append(dist_list['KDE'][-1])

        full_list.append(dist_list)



        # print('Берем корень в после каждого графика:')
        # print('GAN:', dist_list2['GAN'][-1])
        # print('KDE:', dist_list2['KDE'][-1])

    print('Итоговые метрики:')
    print(all_gan_list)
    print(all_kde_list)
    print('full_list:')
    print(full_list)
    x = list(range(2, 11))
    import matplotlib.pyplot as plt
    import numpy as np
    plt.clf()
    plt.plot(x, all_gan_list[1:], label = "GAN")
    plt.plot(x, all_kde_list[1:], label = "KDE")
    plt.xticks(np.arange(min(x), max(x) + 1, 1.0))
    plt.legend()
    # plt.show()
    plt.savefig('GAN_vs_KDE.pdf')
