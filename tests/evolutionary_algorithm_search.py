
class BlockSearcher(object):

    def __init__(self,
                 block_num = 21):
        self.block_num = block_num
        self.total_blocks = ['down_blocks_0_motion_modules_0', 'down_blocks_0_motion_modules_1',
                             'down_blocks_1_motion_modules_0', 'down_blocks_1_motion_modules_1',
                             'down_blocks_2_motion_modules_0', 'down_blocks_2_motion_modules_1',
                             'down_blocks_3_motion_modules_0', 'down_blocks_3_motion_modules_1',
                             "mid_block_motion_modules_0",
                             'up_blocks_0_motion_modules_0', 'up_blocks_0_motion_modules_1',
                             'up_blocks_0_motion_modules_2',
                             'up_blocks_1_motion_modules_0', 'up_blocks_1_motion_modules_1',
                             'up_blocks_1_motion_modules_2',
                             'up_blocks_2_motion_modules_0', 'up_blocks_2_motion_modules_1',
                             'up_blocks_2_motion_modules_2',
                             'up_blocks_3_motion_modules_0', 'up_blocks_3_motion_modules_1',
                             'up_blocks_3_motion_modules_2', ]
        self.candidates = []

    def _set_using_block_num(self):
        assert self.block_num > 0, "block_num should be greater than 0"
        self.block_num = self.block_num - 1

    """
    def is_legal_before_search(self, cand):
        cand = eval(cand)
        cand = sorted(cand)
        cand = str(cand)
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand] # self.vis_dict[candidate1] = {"fid" : XX, "visited" : True} 은 info 이고 이 info 의 fid score 을 저장해 둔다.
        if 'visited' in info:
            logging.info('cand: {} has visited!'.format(cand))
            return False
        info['fid'] = self.get_cand_fid(opt=self.opt, cand=eval(cand))
        logging.info('cand: {}, fid: {}'.format(cand, info['fid']))
        info['visited'] = True
        return True

    def is_legal(self, cand):
        cand = eval(cand)
        cand = sorted(cand)
        cand = str(cand)
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            logging.info('cand: {} has visited!'.format(cand))
            return False
        # if self.RandomForestClassifier.predict_proba(np.asarray(eval(cand), dtype='float')[None, :])[0,1] < self.thres: # 拒绝
        #     logging.info('cand: {} is not legal.'.format(cand))
        #     return False
        info['fid'] = self.get_cand_fid(opt=self.opt, cand=eval(cand))
        logging.info('cand: {}, fid: {}'.format(cand, info['fid']))

        info['visited'] = True
        return True

    def get_random_before_search(self, num):
        logging.info('random select ........')
        while len(self.candidates) < num:
            if self.opt.dpm_solver:
                cand = self.sample_active_subnet_dpm()
            else:
                cand = self.sample_active_subnet()
            cand = sorted(cand)
            cand = str(cand)
            if not self.is_legal_before_search(cand): # vis_dict update
                continue
            self.candidates.append(cand)
            logging.info('random {}/{}'.format(len(self.candidates), num))
        logging.info('random_num = {}'.format(len(self.candidates)))

    def get_random(self, num):
        logging.info('random select ........')
        while len(self.candidates) < num:
            if self.opt.dpm_solver:
                cand = self.sample_active_subnet_dpm()
            else:
                cand = self.sample_active_subnet()
            cand = sorted(cand)
            cand = str(cand)
            if not self.is_legal(cand): # get score and there is no
                continue
            # cand
            self.candidates.append(cand)
            logging.info('random {}/{}'.format(len(self.candidates), num))
        logging.info('random_num = {}'.format(len(self.candidates)))

    def get_cross(self, k, cross_num):
        assert k in self.keep_top_k
        logging.info('cross ......')
        res = []
        max_iters = cross_num * 10

        def random_cross():
            cand1 = choice(self.keep_top_k[k]) # what is difference between choice and random.choice
            cand2 = choice(self.keep_top_k[k]) # to make cross mutation ...

            new_cand = []
            cand1 = eval(cand1)
            cand2 = eval(cand2)
            # isn't cand1 and cand2 same ?
            for i in range(len(cand1)):
                if np.random.random_sample() < 0.5:
                    new_cand.append(cand1[i])
                else:
                    new_cand.append(cand2[i])
            return new_cand

        while len(res) < cross_num and max_iters > 0:
            max_iters -= 1
            cand = random_cross()
            cand = sorted(cand)
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            logging.info('cross {}/{}'.format(len(res), cross_num))

        logging.info('cross_num = {}'.format(len(res)))
        return res

    def get_mutation(self,
                     k,            # 10
                     mutation_num, # 25
                     m_prob):      # .1
        assert k in self.keep_top_k # k = 10
        logging.info('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10 # 250
        def random_func(): # get new condidate timeseries
            cand = choice(self.keep_top_k[k]) # [candidate1, candidate2, ..., candidate10]
            cand = eval(cand) # randomly selected time candidate
            candidates = []
            for i in range(self.sampler.ddpm_num_timesteps):
                if i not in cand:
                    candidates.append(i) # make new candidate
            for i in range(len(cand)):
                if np.random.random_sample() < m_prob:
                    new_c = random.choice(candidates)
                    new_index = candidates.index(new_c)
                    del (candidates[new_index])
                    cand[i] = new_c
                    if len(candidates) == 0:  # cand 的长度小于 candidates 的长度
                        break
            return cand

        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            # random_func() 은 mutation 을 수행하는 함수 # make new candidate
            cand = random_func()
            cand = sorted(cand)
            cand = str(cand)
            # get fid and the visited?
            if not self.is_legal(cand):
                continue
            res.append(cand)
            logging.info('mutation {}/{}'.format(len(res), mutation_num))
        logging.info('mutation_num = {}'.format(len(res)))
        return res



    def get_mutation_dpm(self, k, mutation_num, m_prob):
        assert k in self.keep_top_k
        logging.info('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = choice(self.keep_top_k[k])
            cand = eval(cand)

            candidates = []
            for i in self.dpm_params['full_timesteps']:
                if i not in cand:
                    candidates.append(i)

            for i in range(len(cand)):
                if np.random.random_sample() < m_prob:
                    new_c = random.choice(candidates)
                    new_index = candidates.index(new_c)
                    del (candidates[new_index])
                    cand[i] = new_c
                    if len(candidates) == 0:  # cand 的长度小于 candidates 的长度
                        break

            return cand

        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = random_func()
            cand = sorted(cand)
            cand = str(cand)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            logging.info('mutation {}/{}'.format(len(res), mutation_num))

        logging.info('mutation_num = {}'.format(len(res)))
        return res

    def mutate_init_x(self,
                      x0, # reference timesteps
                      mutation_num, # mutation_num is 24
                      m_prob):      # probability of mutation = 10%
        logging.info('mutation x0 ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10 # 240
        # ------------------------------------------------------------------------------------------
        def random_func():
            # make mutation with randomly changing
            cand = eval(x0) # eval means to list
            candidates = []
            for i in range(self.sampler.ddpm_num_timesteps):
                # ddpm_num_timesteps = 1000
                # i = 0, 1, ..., 999
                print(f'self.sampler.ddpm_num_timesteps = {self.sampler.ddpm_num_timesteps} | i = {i}')
                if i not in cand: # if i not in cand, put i in candidates
                    # therefore, in the candidates, except 1,251,501, 751, others are all in candidates
                    candidates.append(i)
            for i in range(len(cand)):
                # i = 0,1,2,3
                if np.random.random_sample() < m_prob: # if probability is less than 0.1
                    # randomly select c
                    new_c = random.choice(candidates)
                    # get the index in candidates list
                    new_index = candidates.index(new_c)
                    del (candidates[new_index])
                    #
                    cand[i] = new_c
                    if len(candidates) == 0:  # cand 的长度小于 candidates 的长度
                        break
            return cand
        # ------------------------------------------------------------------------------------------

        while len(res) < mutation_num and max_iters > 0: # while len(res) less than 24 and mx_iters > 0 :
            max_iters -= 1
            cand = random_func()
            print(f' [repeating] random funced cand = {cand} (original reference was {x0})')
            cand = sorted(cand)
            cand = str(cand)
            if not self.is_legal_before_search(cand):
                continue
            res.append(cand)
            logging.info('mutation x0 {}/{}'.format(len(res), mutation_num))
        logging.info('mutation_num = {}'.format(len(res)))
        # res is mutated 4 timesteps
        return res

    def mutate_init_x_dpm(self, x0, mutation_num, m_prob):
        logging.info('mutation x0 ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = x0
            cand = eval(cand)

            candidates = []
            for i in self.dpm_params['full_timesteps']:
                if i not in cand:
                    candidates.append(i)

            for i in range(len(cand)):
                if np.random.random_sample() < m_prob:
                    new_c = random.choice(candidates)
                    new_index = candidates.index(new_c)
                    del (candidates[new_index])
                    cand[i] = new_c
                    if len(candidates) == 0:  # cand 的长度小于 candidates 的长度
                        break

            return cand

        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = random_func()
            cand = sorted(cand)
            cand = str(cand)
            if not self.is_legal_before_search(cand):
                continue
            res.append(cand)
            logging.info('mutation x0 {}/{}'.format(len(res), mutation_num))

        logging.info('mutation_num = {}'.format(len(res)))
        return res

    def sample_active_subnet(self):
        original_num_steps = self.sampler.ddpm_num_timesteps
        use_timestep = [i for i in range(original_num_steps)]
        random.shuffle(use_timestep)
        use_timestep = use_timestep[:self.time_step]
        # use_timestep = [use_timestep[i] + 1 for i in range(len(use_timestep))]
        return use_timestep

    def sample_active_subnet_dpm(self):
        use_timestep = copy.deepcopy(self.dpm_params['full_timesteps'])
        random.shuffle(use_timestep)
        use_timestep = use_timestep[:self.time_step + 1]
        # use_timestep = [use_timestep[i] + 1 for i in range(len(use_timestep))]
        return use_timestep

    def get_cand_fid(self, cand=None, opt=None, device='cuda'):
        t1 = time.time()
        start_code = None
        if opt.fixed_code:
            start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
        precision_scope = autocast if opt.precision == "autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    t1 = time.time()
                    all_samples = list()
                    for itr, batch in enumerate(self.dataloader_info['validation_loader']):
                        prompts = batch['text']
                        uc = None
                        if opt.scale != 1.0:
                            uc = self.model.get_learned_conditioning(self.batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = self.model.get_learned_conditioning(prompts)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        sampled_timestep = np.array(cand)
                        samples_ddim, _ = self.sampler.sample(S=opt.time_step,
                                                              conditioning=c,
                                                              batch_size=opt.n_samples,
                                                              shape=shape,
                                                              verbose=False,
                                                              unconditional_guidance_scale=opt.scale,
                                                              unconditional_conditioning=uc,
                                                              eta=opt.ddim_eta,
                                                              x_T=start_code,
                                                              sampled_timestep=sampled_timestep)
                        x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                        # x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)
                        x_checked_image = x_samples_ddim
                        x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)
                        for x_sample in x_checked_image_torch:
                            all_samples.append(x_sample.cpu().numpy())
                        logging.info('samples: ' + str(len(all_samples)))
                        if len(all_samples) > self.num_samples:
                            logging.info('samples: ' + str(len(all_samples)))
                            break
        sample_time = time.time() - t1
        t1 = time.time()
        all_samples = np.array(all_samples)
        all_samples = torch.Tensor(all_samples)
        fid = calculate_fid(data1=all_samples, ref_mu=self.ref_mu, ref_sigma=self.ref_sigma, batch_size=320, dims=2048, device='cuda')
        logging.info('FID: ' + str(fid))
        fid_time = time.time() - t1
        logging.info('sample_time: ' + str(sample_time) + ', fid_time: ' + str(fid_time))
        return fid
    """
    def select_blocks(self) :
        while True :
            random_idx = np.random.choice(len(self.total_blocks), self.block_num, replace=False).tolist() # ex) [1,2,3]
            random_idx = sorted(random_idx)
            if random_idx not in self.conditions:
                self.candidates.append(random_idx)
            else :
                continue

    def search(self) :

        # [1] set the number of blocks
        self._set_using_block_num() # self.block_num = 20
        logging.info(f'block num {self.block_num} experiment start')

        # [2.1]
        # select self.bock_num blocks in total_blocks
        self.select_blocks() # updated self.candidates

    def reset (self):

        self.candidates = []

def main() :

    total_block_num = 21
    searcher = BlockSearcher(block_num=total_block_num)
    for i in range(total_block_num) :
        searcher.search()
        candidates = searcher.candidates
        print(candidates)
        searcher.reset
    # [2]


if __name__ == '__main__' :
    main()
