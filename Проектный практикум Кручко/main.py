import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import time
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

print("\n" + "="*70)
print(" " * 20 + "АНАЛИЗ ИГРОВОЙ СРЕДЫ")
print("="*70)

env = gym.make("CartPole-v1")

print(f"\nХарактеристики среды:")
print(f"Состояние: {env.observation_space} (4 параметра)")
print(f"Действий: {env.action_space.n} (0=влево, 1=вправо)")
print(f"Макс. шагов для победы: {env.spec.max_episode_steps}")

state, _ = env.reset()
print(f"\nПример состояния (входные данные нейросети):")
print(f"Позиция тележки: {state[0]:6.3f}")
print(f"Скорость тележки: {state[1]:6.3f}")
print(f"Угол шеста: {state[2]:6.3f} рад")
print(f"Угловая скорость: {state[3]:6.3f} рад/с")

env.close()

class SimpleNetwork(nn.Module):
    def __init__(self, input_dim=4, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.net(x)


class ComplexNetwork(nn.Module):
    def __init__(self, input_dim=4, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.net(x)

model_simple = SimpleNetwork()
model_complex = ComplexNetwork()

print("\n" + "="*70)
print(" " * 20 + "АРХИТЕКТУРЫ НЕЙРОСЕТЕЙ")
print("="*70)

for name, model in [("ПРОСТАЯ сеть (64-32)", model_simple), 
                    ("СЛОЖНАЯ сеть (128-64-32-16)", model_complex)]:
    params = sum(p.numel() for p in model.parameters())
    print(f"\n{name}:")
    print(f" Параметров: {params:,}")
    print(f" Структура: {model}")

class REINFORCEAgent:
    def __init__(self, model, lr=0.0005, gamma=0.99):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.saved_log_probs = []
        self.saved_rewards = []
    
    def select_action(self, state):
        #Выбор действия с сохранением log вероятности
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.model(state_tensor)
        
        #Стохастический выбор по политике
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        
        return action.item()
    
    def compute_returns(self, rewards):
        #Дисконтированные возвраты с нормализацией
        returns = []
        R = 0
        
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, dtype=torch.float32)
        
        #Нормализация для уменьшения дисперсии
        if len(returns) > 1 and returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns
    
    def train_episode(self, env):
        #Обучение на одном эпизоде
        state, _ = env.reset()
        self.saved_log_probs = []
        self.saved_rewards = []
        done = False
        
        while not done:
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            self.saved_rewards.append(reward)
            state = next_state
        
        #Вычисляем returns
        returns = self.compute_returns(self.saved_rewards)
        
        #Функция потерь
        policy_loss = []
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        #Обновление весов
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        
        #Градиентное клиппинг (важно для стабильности)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return sum(self.saved_rewards), len(self.saved_rewards)

def train_model(model, episodes=600, lr=0.0005, gamma=0.99, print_every=100):
    #Обучение модели
    env = gym.make("CartPole-v1")
    agent = REINFORCEAgent(model, lr=lr, gamma=gamma)
    
    rewards_history = []
    steps_history = []
    times_history = []
    best_reward = 0
    
    print(f"\nНачало обучения ({episodes} эпизодов)...")
    
    for ep in range(1, episodes + 1):
        start_time = time.time()
        total_reward, steps = agent.train_episode(env)
        ep_time = time.time() - start_time
        
        rewards_history.append(total_reward)
        steps_history.append(steps)
        times_history.append(ep_time)
        
        if total_reward > best_reward:
            best_reward = total_reward
        
        if ep % print_every == 0:
            avg_r = np.mean(rewards_history[-print_every:])
            avg_s = np.mean(steps_history[-print_every:])
            print(f"  Эпизод {ep:4d}: ср.награда={avg_r:6.1f}, "
                  f"ср.шаги={avg_s:6.1f}, лучшая={best_reward:4.0f}")
    
    env.close()
    return agent, rewards_history, steps_history, times_history


def test_model(model, episodes=200):
    #Тестирование модели (ВАЖНО: model.eval() отключает Dropout)
    #КРИТИЧЕСКИ ВАЖНО: отключаем режим обучения
    model.eval()
    
    env = gym.make("CartPole-v1")
    test_rewards = []
    successes = 0
    inference_times = []
    
    print(f"\nТестирование ({episodes} эпизодов)...")
    
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            t0 = time.time()
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            with torch.no_grad():
                probs = model(state_tensor)
            
            #Детерминированный выход
            action = torch.argmax(probs, dim=1).item()
            inference_times.append(time.time() - t0)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state
        
        test_rewards.append(total_reward)
        if total_reward >= 500:
            successes += 1
        
        #Показываем прогресс
        if (episode + 1) % 50 == 0:
            current_rate = successes / (episode + 1) * 100
            print(f"  Тест {episode + 1:3d}/{episodes}: "
                  f"успешность={current_rate:.1f}%")
    
    env.close()
    
    #Возвращаем модель в режим обучения (на всякий случай)
    model.train()
    
    return {
        'avg_reward': np.mean(test_rewards),
        'std_reward': np.std(test_rewards),
        'success_rate': successes / episodes * 100,
        'avg_inference_ms': np.mean(inference_times) * 1000,
        'min_reward': np.min(test_rewards),
        'max_reward': np.max(test_rewards),
        'rewards': test_rewards
    }

print("\n" + "="*70)
print(" " * 20 + "НАЧАЛО ЭКСПЕРИМЕНТА")
print("="*70)

#Параметры обучения
EPISODES = 600
LR = 0.0005
GAMMA = 0.99

#Обучаем простую сеть
print("\n" + "-"*70)
print("ОБУЧЕНИЕ 1/2: ПРОСТАЯ сеть (64-32)")
print("-"*70)
agent_simple, rewards_simple, steps_simple, times_simple = train_model(
    model_simple, episodes=EPISODES, lr=LR, gamma=GAMMA
)

#Обучаем сложную сеть
print("\n" + "-"*70)
print("ОБУЧЕНИЕ 2/2: СЛОЖНАЯ сеть (128-64-32-16)")
print("-"*70)
agent_complex, rewards_complex, steps_complex, times_complex = train_model(
    model_complex, episodes=EPISODES, lr=LR, gamma=GAMMA
)

print("\n" + "="*70)
print(" " * 25 + "ТЕСТИРОВАНИЕ МОДЕЛЕЙ")
print("="*70)

print("\nТестирование ПРОСТОЙ сети...")
test_simple = test_model(model_simple, episodes=200)

print("\nТестирование СЛОЖНОЙ сети...")
test_complex = test_model(model_complex, episodes=200)

print("\n" + "="*70)
print(" " * 25 + "ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
print("="*70)

#Функция для скользящего среднего
def moving_avg(data, window=20):
    return [np.mean(data[max(0,i-window):i+1]) for i in range(len(data))]

fig = plt.figure(figsize=(16, 10))

# 1. Кривые обучения (сравнение)
ax1 = plt.subplot(2, 3, 1)
ax1.plot(moving_avg(rewards_simple), 'b-', linewidth=2, label='Простая сеть')
ax1.plot(moving_avg(rewards_complex), 'r-', linewidth=2, label='Сложная сеть')
ax1.axhline(500, color='g', linestyle='--', alpha=0.5, label='Максимум')
ax1.set_xlabel('Эпизод')
ax1.set_ylabel('Награда (сглаженная)')
ax1.set_title('Сравнение кривых обучения')
ax1.legend()
ax1.grid(alpha=0.3)
ax1.set_ylim(0, 520)

# 2. Успешность на тесте
ax2 = plt.subplot(2, 3, 2)
models = ['Простая\n(64-32)', 'Сложная\n(128-64-32-16)']
success_rates = [test_simple['success_rate'], test_complex['success_rate']]
colors = ['steelblue', 'coral']
bars = ax2.bar(models, success_rates, color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylabel('Успешность (%)')
ax2.set_title('Успешность на тесте (200 эпизодов)')
ax2.set_ylim(0, 105)
ax2.grid(axis='y', alpha=0.3)

#Добавляем значения на столбцы
for bar, rate in zip(bars, success_rates):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

# 3. Средняя награда с ошибками
ax3 = plt.subplot(2, 3, 3)
avg_rewards = [test_simple['avg_reward'], test_complex['avg_reward']]
std_rewards = [test_simple['std_reward'], test_complex['std_reward']]
bars = ax3.bar(models, avg_rewards, yerr=std_rewards, capsize=8, 
               color=colors, alpha=0.7, edgecolor='black')
ax3.axhline(500, color='g', linestyle='--', alpha=0.5, label='Максимум')
ax3.set_ylabel('Средняя награда')
ax3.set_title('Средняя награда (± стандартное отклонение)')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

for bar, avg in zip(bars, avg_rewards):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f'{avg:.1f}', ha='center', fontweight='bold')



# 5. Время обучения
ax5 = plt.subplot(2, 3, 5)
train_times = [np.sum(times_simple), np.sum(times_complex)]
bars = ax5.bar(models, train_times, color=colors, alpha=0.7, edgecolor='black')
ax5.set_ylabel('Время (секунды)')
ax5.set_title('Общее время обучения')
ax5.grid(axis='y', alpha=0.3)

for bar, time in zip(bars, train_times):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f'{time:.1f}с', ha='center', fontweight='bold')

# 6. Время вывода
ax6 = plt.subplot(2, 3, 6)
inf_times = [test_simple['avg_inference_ms'], test_complex['avg_inference_ms']]
bars = ax6.bar(models, inf_times, color=colors, alpha=0.7, edgecolor='black')
ax6.set_ylabel('Время (миллисекунды)')
ax6.set_title('Среднее время принятия решения')
ax6.grid(axis='y', alpha=0.3)

for bar, time in zip(bars, inf_times):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{time:.3f}мс', ha='center', fontweight='bold')

plt.suptitle('Сравнение архитектур нейросетей на задаче CartPole-v1', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# Дополнительный график: сырые данные обучения
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(rewards_simple, 'b-', alpha=0.3, linewidth=0.5, label='Сырые данные')
ax1.plot(moving_avg(rewards_simple), 'b-', linewidth=2, label='Сглаженное (окно=20)')
ax1.axhline(500, color='g', linestyle='--', alpha=0.5)
ax1.set_xlabel('Эпизод')
ax1.set_ylabel('Награда')
ax1.set_title('Простая сеть (64-32) - процесс обучения')
ax1.legend()
ax1.grid(alpha=0.3)
ax1.set_ylim(0, 520)

ax2.plot(rewards_complex, 'r-', alpha=0.3, linewidth=0.5, label='Сырые данные')
ax2.plot(moving_avg(rewards_complex), 'r-', linewidth=2, label='Сглаженное (окно=20)')
ax2.axhline(500, color='g', linestyle='--', alpha=0.5)
ax2.set_xlabel('Эпизод')
ax2.set_ylabel('Награда')
ax2.set_title('Сложная сеть (128-64-32-16) - процесс обучения')
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_ylim(0, 520)

plt.suptitle('Детальный анализ обучения двух архитектур', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n" + "="*80)
print(" " * 30 + "ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
print("="*80)

print(f"\n{'Метрика':<35} {'Простая сеть (64-32)':<22} {'Сложная сеть (128-64-32-16)':<22}")
print("-"*80)

print(f"{'Успешность на тесте (%)':<35} "
      f"{test_simple['success_rate']:>18.1f}%          "
      f"{test_complex['success_rate']:>18.1f}%")

print(f"{'Средняя награда':<35} "
      f"{test_simple['avg_reward']:>18.1f} ± {test_simple['std_reward']:<5.1f}    "
      f"{test_complex['avg_reward']:>18.1f} ± {test_complex['std_reward']:<5.1f}")

print(f"{'Мин / Макс награда':<35} "
      f"{test_simple['min_reward']:>8.0f} / {test_simple['max_reward']:<8.0f}      "
      f"{test_complex['min_reward']:>8.0f} / {test_complex['max_reward']:<8.0f}")

print(f"{'Время обучения (всего)':<35} "
      f"{np.sum(times_simple):>18.2f}с          "
      f"{np.sum(times_complex):>18.2f}с")

print(f"{'Время обучения (среднее за эпизод)':<35} "
      f"{np.mean(times_simple):>18.3f}с          "
      f"{np.mean(times_complex):>18.3f}с")

print(f"{'Время вывода (среднее)':<35} "
      f"{test_simple['avg_inference_ms']:>18.3f}мс        "
      f"{test_complex['avg_inference_ms']:>18.3f}мс")

print(f"{'Количество параметров':<35} "
      f"{sum(p.numel() for p in model_simple.parameters()):>18,}          "
      f"{sum(p.numel() for p in model_complex.parameters()):>18,}")

# ==================== ШАГ 9: АНАЛИЗ СТАБИЛЬНОСТИ ====================

print("\n" + "="*80)
print(" " * 25 + "АНАЛИЗ СТАБИЛЬНОСТИ И РАЗБРОСА РЕЗУЛЬТАТОВ")
print("="*80)

# Коэффициент вариации (CV = std/mean * 100%)
cv_simple = (test_simple['std_reward'] / test_simple['avg_reward']) * 100
cv_complex = (test_complex['std_reward'] / test_complex['avg_reward']) * 100

print(f"\nКоэффициент вариации (CV):")
print(f"Простая сеть: {cv_simple:.2f}% {'(Стабильная)' if cv_simple < 5 else '(Нестабильная)'}")
print(f"Сложная сеть: {cv_complex:.2f}% {'(Стабильная)' if cv_complex < 5 else '(Нестабильная)'}")

print(f"\nРазброс результатов на тесте (200 эпизодов):")
print(f"Простая сеть: от {test_simple['min_reward']:.0f} до {test_simple['max_reward']:.0f} "
      f"(размах = {test_simple['max_reward'] - test_simple['min_reward']:.0f})")
print(f"Сложная сеть: от {test_complex['min_reward']:.0f} до {test_complex['max_reward']:.0f} "
      f"(размах = {test_complex['max_reward'] - test_complex['min_reward']:.0f})")

print("\n" + "="*80)
print(" " * 30 + "ВЫВОДЫ И РЕКОМЕНДАЦИИ")
print("="*80)

# Определяем победителя по разным метрикам
if test_simple['success_rate'] > test_complex['success_rate']:
    winner_success = "ПРОСТАЯ сеть"
else:
    winner_success = "СЛОЖНАЯ сеть"

if np.sum(times_simple) < np.sum(times_complex):
    winner_time = "ПРОСТАЯ сеть"
else:
    winner_time = "СЛОЖНАЯ сеть"

if test_simple['avg_inference_ms'] < test_complex['avg_inference_ms']:
    winner_speed = "ПРОСТАЯ сеть"
else:
    winner_speed = "СЛОЖНАЯ сеть"

print(f"\nПОБЕДИТЕЛИ ПО МЕТРИКАМ:")
print(f"По успешности: {winner_success} ({max(success_rates):.1f}%)")
print(f"По времени обучения: {winner_time}")
print(f"По скорости вывода: {winner_speed}")

print(f"\nОСНОВНЫЕ НАБЛЮДЕНИЯ:")
if test_simple['success_rate'] > test_complex['success_rate'] + 10:
    print(f"Простая сеть значительно лучше (разница > 10%)")
    print(f"Сложная сеть переобучается или не успевает обучиться")
elif test_complex['success_rate'] > test_simple['success_rate'] + 10:
    print(f"Сложная сеть значительно лучше (разница > 10%)")
    print(f"Дополнительные слои помогают обобщать")
else:
    print(f"Обе сети показывают сопоставимые результаты")
    print(f"Разница в успешности: {abs(test_simple['success_rate'] - test_complex['success_rate']):.1f}%")

print("\n" + "="*80)
print(" " * 30 + "ЭКСПЕРИМЕНТ ЗАВЕРШЕН")
print("="*80)
print("\nОбе модели обучены и протестированы!")
print("Все графики сохранены и отображены")
print("Результаты показывают РАЗНЫЕ метрики для двух архитектур")