import os, json, ast

def main() :

    def get_info (dir) :

        with open(dir, 'r') as f:
            latent = f.readlines()
        arch_list, fvd_list = [], []
        for line in latent:
            line = line.strip()
            architecture, info = line.split(' : ')
            architecture = tuple(ast.literal_eval(architecture.split('Architecture')[-1]))
            info = ast.literal_eval(info)
            fvd = info['fvd']
            arch_list.append(architecture)
            fvd_list.append(fvd)
        return arch_list, fvd_list

    latent_dir = 'latent_logs.txt'
    latent_arcs, latent_fvd = get_info(latent_dir)

    def make_text_file(arcs, fvd, name) :
        with open(name, 'a') as f:
            for arc, fvd in zip(arcs, fvd):
                f.write(f'{arc} : {fvd}\n')
    # make file
    latent_excel_file = 'latent_excel.txt'
    make_text_file(latent_arcs, latent_fvd, latent_excel_file)


    pixel_dir = 'pixel_logs.txt'
    pixel_arcs, pixel_fvd = get_info(pixel_dir)
    pixel_excel_file = 'pixel_excel.txt'
    make_text_file(pixel_arcs, pixel_fvd, pixel_excel_file)


    for pixel_arc in pixel_arcs:
        if pixel_arc in latent_arcs:
            print(f'Found a match: {pixel_arc}')

if __name__ == '__main__':
    main()