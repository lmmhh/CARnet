#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import argparse
import os

import pandas as pd
import torch.utils.data
from torch.utils.data import DataLoader
from skimage.measure import compare_mse, compare_ssim
from utils.data_utils import TestDatasetFromFolder, ValDatasetFromFolder
import importlib
from libtiff import TIFF
import numpy as np
from utils.LS_index import LSIndex

parser = argparse.ArgumentParser(description='Test JPEG-LS Models')
parser.add_argument('--model', default='CARnet', type=str, help='model type')
parser.add_argument('--data', default='10', type=str, help='data type')
parser.add_argument('--near', default='LS-N12', type=str, help='LS type')
parser.add_argument('--ckpt_path', default='./checkpoint', type=str, help='checkpoint path')
parser.add_argument('--data_path', default='./data', type=str, help='data path')
parser.add_argument('--out_path', default='./test_out', type=str, help='test out path')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def mse2psnr(err, max_value=1.):
    return 10 * np.log10((max_value ** 2) / err)


if __name__ == '__main__':
    opt = parser.parse_args()
    if not os.path.isdir(opt.out_path):
        os.mkdir(opt.out_path)
    model_module = importlib.import_module('models.' + opt.model)

    Dataset_Ori = os.path.join(os.path.join(opt.data_path, opt.data), 'Ori')
    Dataset_LS = os.path.join(os.path.join(opt.data_path, opt.data), opt.near)
    ckpt_path = os.path.join(opt.ckpt_path, opt.data + '_' + opt.near + '.pth')

    test_out_path = os.path.join(opt.out_path, opt.model + '_' + opt.data + '_' + opt.near)
    if not os.path.isdir(test_out_path):
        os.mkdir(test_out_path)

    if opt.data == '10':
        data_range = 1023.
    elif opt.data == '11':
        data_range = 2047.
    elif opt.data == '12':
        data_range = 4095.
    else:
        raise NotImplementedError('Needs to right data type.')

    test_set = ValDatasetFromFolder(os.path.join(Dataset_Ori, 'crop256'),
                                    os.path.join(Dataset_LS, 'crop256'), data_range=data_range)

    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)

    model = model_module.get_model_spec()
    model.load_state_dict(torch.load(ckpt_path))

    if torch.cuda.is_available():
        model.cuda()
    else:
        print('test with cpu')

    results = {'image': [], 'test_mse': [], 'test_ssim': [], 'test_ssimls': [], 'test_psnrls': [], 'ori_mse': [],
               'ori_ssim': [], 'ori_ssimls': [], 'ori_psnrls': []}

    with torch.no_grad():
        total_c_mse = 0
        total_p_mse = 0
        total_c_ssim = 0
        total_p_ssim = 0
        total_c_ssimls = 0
        total_p_ssimls = 0
        total_c_psnrls = 0
        total_p_psnrls = 0
        tl = len(test_loader)
        model.eval()
        for image_name, test_lr, test_hr, run_length, near in test_loader:
            if torch.cuda.is_available():
                lr = test_lr.cuda()
                r = run_length.cuda()
                out = model(lr)
                sr = out.cpu().numpy()
            else:
                out = model(test_lr)
                sr = out.numpy()

            sr = np.multiply(np.clip(np.squeeze(sr), 0, 1), data_range).astype(np.uint16)
            lr = np.multiply(np.squeeze(test_lr.numpy()), data_range).astype(np.uint16)
            hr = np.multiply(np.squeeze(test_hr.numpy()), data_range).astype(np.uint16)

            index = LSIndex(win_size=11)
            c_mse = compare_mse(hr, lr)
            p_mse = compare_mse(hr, sr)
            c_ssim = compare_ssim(hr, lr, win_size=11, data_range=data_range)
            p_ssim = compare_ssim(hr, sr, win_size=11, data_range=data_range)
            c_ssimls = index.ssim(hr, lr, win_size=11, data_range=data_range)
            p_ssimls = index.ssim(hr, sr, win_size=11, data_range=data_range)
            c_psnrls = index.psnr(hr, lr, data_range=data_range)
            p_psnrls = index.psnr(hr, sr, data_range=data_range)
            results['image'].append(image_name[0])
            results['test_mse'].append(p_mse)
            results['test_ssim'].append(p_ssim)
            results['test_ssimls'].append(p_ssimls)
            results['test_psnrls'].append(p_psnrls)
            results['ori_mse'].append(c_mse)
            results['ori_ssim'].append(c_ssim)
            results['ori_ssimls'].append(c_ssimls)
            results['ori_psnrls'].append(c_psnrls)
            total_c_mse += c_mse
            total_p_mse += p_mse
            total_c_ssim += c_ssim
            total_p_ssim += p_ssim
            total_c_ssimls += c_ssimls
            total_p_ssimls += p_ssimls
            total_c_psnrls += c_psnrls
            total_p_psnrls += p_psnrls
            image_out_path = os.path.join(test_out_path, image_name[0].split('.')[0]+'.tif')
            tiff = TIFF.open(image_out_path, mode='w')
            tiff.write_image(sr)
            tiff.close()

        total_c_mse /= tl
        total_p_mse /= tl
        total_c_ssim /= tl
        total_p_ssim /= tl
        total_c_ssimls /= tl
        total_p_ssimls /= tl
        total_c_psnrls /= tl
        total_p_psnrls /= tl
        print('The average ' + 'MSE: %.12f -> %.12f, PSNR: %.8f -> %.8f, SSIM: %.8f -> %.8f, SSIMLS: %.8f -> %.8f, PSNRLS: %.8f -> %.8f\n' %
              (total_c_mse, total_p_mse,
               mse2psnr(total_c_mse, data_range), mse2psnr(total_p_mse, data_range),
               total_c_ssim, total_p_ssim, total_c_ssimls, total_p_ssimls, total_c_psnrls, total_p_psnrls))

        out_result_path = test_out_path + opt.model + '_' + opt.data + '_' + opt.near + '_results.csv'
        data_frame = pd.DataFrame(data={'Image': results['image'], 'Test MSE': results['test_mse'],
                                        'Test SSIM': results['test_ssim'], 'Test SSIMLS': results['test_ssimls'],
                                        'Test PSNRLS': results['test_psnrls']},
                                  index=range(1, tl+1))
        data_frame.to_csv(out_result_path, index_label='index')
