import time
import numpy as np
import torch

from logger.saver import Saver
from logger import utils


def test(args, model, loss_func, loader_test, saver):
    """
    Reports:
        - Generated audio
        - Total loss
        - Inference time / RTF
    """
    print(' [*] testing...')
    model.eval()

    # losses
    test_loss = 0.
    
    # intialization
    num_batches = len(loader_test)
    rtf_all = []
    
    # run
    with torch.no_grad():
        for bidx, data in enumerate(loader_test):
            fn = data['name'][0]
            print('--------')
            print(f'{bidx}/{num_batches} - {fn}')

            # unpack data
            for k in data.keys():
                if k != 'name':
                    data[k] = data[k].to(args.device)
            print('>>', data['name'][0])
            audio_gt = data['audio']
            # Reconstruction forward
            st_time = time.time()
            audio_pred, _, _ = model(data['units'], data['f0'], data['volume'], data['spk_id'])
            ed_time = time.time()

            # crop
            min_len = np.min([audio_pred.shape[1], audio_gt.shape[1]])
            audio_pred = audio_pred[:, :min_len]
            audio_gt   =   audio_gt[:, :min_len]

            # RTF
            run_time = ed_time - st_time
            song_time = audio_gt.shape[-1] / args.data.sampling_rate
            rtf = run_time / song_time
            print(f'RTF: {rtf}  | {run_time} / {song_time}')
            rtf_all.append(rtf)
           
            # loss
            loss = loss_func(audio_pred, audio_gt)
            test_loss += loss.item()

            # log
            saver.log_audio({fn+'/gt.wav': audio_gt, fn+'/pred.wav': audio_pred})
            
    # report
    test_loss /= num_batches
    
    # check
    print(' [test_loss] test_loss:', test_loss)
    print(' Real Time Factor', np.mean(rtf_all))
    return test_loss


def train(args, initial_global_step, model, optimizer, loss_func, loader_train, loader_test):
    """
    Args:

        model - Sins | CombSub
    """
    # Init
    ## Saver
    saver = Saver(args, initial_global_step=initial_global_step)
    ## Model size
    saver.log_info('--- model size ---')
    saver.log_info(utils.get_network_paras_amount({'model': model}))
    ## Runner    
    best_loss = np.inf
    num_batches = len(loader_train)
    model.train()

    saver.log_info('======= start training =======')
    for epoch in range(args.train.epochs):
        for batch_idx, data in enumerate(loader_train):
            saver.global_step_increment()

            # unpack data
            for k in data.keys():
                if k != 'name':
                    data[k] = data[k].to(args.device)
            
            # forward/loss/Backward/Optim
            optimizer.zero_grad()
            signal, _, _ = model(data['units'].float(), data['f0'], data['volume'], data['spk_id'], infer=False)
            loss = loss_func(signal, data['audio'])
            loss.backward()
            optimizer.step()

            # log loss
            if saver.global_step % args.train.interval_log == 0:
                saver.log_info(
                    'epoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | loss: {:.3f} | time: {} | step: {}'.format(
                        epoch,
                        batch_idx, num_batches,
                        args.env.expdir,
                        args.train.interval_log/saver.get_interval_time(),
                        loss.item(),
                        saver.get_total_time(),
                        saver.global_step
                ))
                saver.log_value({ 'train/loss': loss.item() })
            
            if saver.global_step % args.train.interval_val == 0:
                # Validation
                test_loss = test(args, model, loss_func, loader_test, saver)
                saver.log_info(' --- <validation> --- \nloss: {:.3f}. '.format(test_loss))
                saver.log_value({ 'validation/loss': test_loss })
                # Save
                # Latest
                saver.save_model(model, optimizer, postfix=f'{saver.global_step}')
                ## Best
                if test_loss < best_loss:
                    saver.log_info(' [V] best model updated.')
                    saver.save_model(model, optimizer, postfix='best')
                    best_loss = test_loss
                model.train()
