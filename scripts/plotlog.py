import argparse
import pandas as pd
import h5py
import json
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('logfiles', type=str, nargs='+')
    parser.add_argument('--fields', type=str, default='trueret,avglen,ent,kl,vf_r2,vf_kl,tdvf_r2,rloss,racc')
    #parser.add_argument('--fields', type=str, default='trueret,bcloss,valloss,valacc')
    parser.add_argument('--noplot', action='store_true')
    parser.add_argument('--plotfile', type=str, default=None)
    parser.add_argument('--range_end', type=int, default=None)
    args = parser.parse_args()

    assert len(set(args.logfiles)) == len(args.logfiles), 'Log files must be unique'

    fields = args.fields.split(',')

    # Load logs from all files
    fname2log = {}
    for fname in args.logfiles:
        with pd.HDFStore(fname, 'r') as f:
            assert fname not in fname2log
            df = f['log']
            df.set_index('iter', inplace=True)
            log_list = []
            for field in fields:
                log_list.append(df.loc[:args.range_end, [field]])
            fname2log[fname] = log_list


    # Print stuff
    if not args.noplot or args.plotfile is not None:
        import matplotlib
        if args.plotfile is not None:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt; plt.style.use('ggplot')

        ax = None
        for fname, logs in fname2log.items():
            for df in logs:
                with pd.option_context('display.max_rows', 9999):
                    print fname
                    print df[-1:]


                field = df.keys()[0]
                if field == 'vf_r2':
                     df['vf_r2'] = np.maximum(0,df['vf_r2'])
                if field == 'trueret':
                     print('maximum avg return: %f at iter %d' % (df['trueret'].max(), df['trueret'].argmax()))

                if ax is None:
                    ax = df.plot(subplots=True, title=fname)
                if not args.noplot:
                    plt.show()
                if args.plotfile is not None:
                    plt.savefig(args.plotfile+'_'+field, bbox_inches='tight', dpi=200)
                ax = None
        


if __name__ == '__main__':
    main()
