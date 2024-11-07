import random


class BlockSearchGA:
    def __init__(self, total_block_num, population_size, min_blocks, max_blocks):
        self.total_block_num = total_block_num  # 총 블록 수
        self.population_size = population_size  # 개체군 크기
        self.min_blocks = min_blocks  # 블록의 최소 개수
        self.max_blocks = max_blocks  # 블록의 최대 개수
        #self.target_block_num = target_block_num  # 목표 블록 수 (초기 개체군에서만 사용) (=10)
        #self.fitness_fn = fitness_fn  # fitness 함수 (적합도 계산 함수)

    def generate_initial_population(self):
        # 초기 개체군을 생성하는 메서드
        population = []
        for _ in range(self.population_size):
            # 임의로 X 개의 블록을 샘플링 (X는 min_blocks와 max_blocks 사이에서 랜덤)
            target_block_num = random.randint(self.min_blocks, self.max_blocks)  # 임의의 블록 수 결정
            architecture = sorted(random.sample(range(self.total_block_num),
                                                k=target_block_num))  # 선택된 블록 수 만큼 샘플링
            population.append(architecture)  # 생성된 아키텍처를 개체군에 추가
        return population

    def evaluate_population(self, population):
        # 각 개체의 fitness 값을 평가하는 메서드
        fitness_scores = []
        for architecture in population:

            # [1] generate video


            # [2] get score from that architecture
            fitness_score = self.fitness_fn(architecture)  # 주어진 fitness 함수로 평가 (block number)
            fitness_scores.append((architecture, fitness_score))

        # fitness 기준으로 내림차순 정렬하면, block 수가 적은 것이 위로 올라가서 block 이 적은 것이 좋게 됨
        return sorted(fitness_scores, key=lambda x: x[1], reverse=False)  # fitness 기준으로 내림차순 정렬 #  m

    def mutate(self, architecture):
        # 돌연변이 연산: 블록의 개수를 변경하고, 블록 인덱스를 임의로 변경
        target_block_num = random.randint(self.min_blocks, self.max_blocks)  # 새로 설정할 블록 수
        architecture = sorted(random.sample(range(self.total_block_num), k=target_block_num))  # 새 블록 샘플링
        return architecture

    def crossover(self, parent1, parent2):
        # 교차 연산: 두 부모 아키텍처의 블록들을 섞어서 새로운 자식 생성
        crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)  # 교차 지점 설정
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]

        # 두 자식 모두 블록의 개수를 임의로 변경하도록 처리
        child1 = self.mutate(child1)
        child2 = self.mutate(child2)

        return child1, child2

    def select_parents(self, population):
        # 부모 선택: 적합도가 높은 상위 50%에서 부모를 선택
        selected_parents = population[:len(population) // 2]
        return selected_parents

    def evolve(self, generations):

        # [1] make initial population
        population = self.generate_initial_population()

        for generation in range(generations):

            print(f"Generation {generation + 1}")

            # 개체군의 fitness 평가
            sorted_population = self.evaluate_population(population)

            # 상위 50% 선택 (선택적 교차 및 돌연변이)
            parents = self.select_parents(sorted_population)
            next_generation = []

            # 교차 및 돌연변이
            for i in range(0, len(parents), 2):
                parent1, _ = parents[i]
                parent2, _ = parents[i + 1] if i + 1 < len(parents) else parents[i]

                child1, child2 = self.crossover(parent1, parent2)
                next_generation.append(child1)
                next_generation.append(child2)

            # 새로운 세대에 추가
            population = next_generation

            # 상위 fitness 출력
            best_architecture, best_fitness = sorted_population[0]
            print(f"Best Architecture: {best_architecture} with Fitness: {best_fitness}")

        # 최종적으로 가장 적합한 개체 반환
        return sorted_population[0]


    # 예시 fitness 함수
    def fitness_fn(self, architecture):
        # 여기서는 간단히 블록 수에 대한 페널티를 부여하는 예시를 듬
        return len(architecture)  # 블록 수에 비례한 간단한 fitness

def main() :

    # 예시로 이 클래스 사용
    ga = BlockSearchGA(
        total_block_num=20,  # 총 20개의 블록
        population_size=10,  # 10개의 개체
        min_blocks=5,  # 최소 5개의 블록
        max_blocks=15, ) # 최대 15개의 블록
    best_architecture = ga.evolve(generations=10)
    print(f"Best Architecture Found: {best_architecture}")

if __name__ == '__main__':
    main()