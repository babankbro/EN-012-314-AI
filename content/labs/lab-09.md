## ข้อมูลทั่วไปของใบงาน

| รายการ | รายละเอียด |
|---|---|
| ชื่อใบงาน | Lab 09: Q-Learning กับเกม Grid World |
| วิชา | EN-012-314 ปัญญาประดิษฐ์ (Artificial Intelligence) — ส่วน Machine Learning |
| สัปดาห์ | 13 |
| งานตาม มคอ.3 | แบบฝึกหัด 8: RL สำหรับ game playing |
| CLO/LLO | CLO4 / LLO4-3 อธิบายหลักการ reinforcement learning และประยุกต์ใช้กับปัญหา game playing เบื้องต้น |
| เครื่องมือ | Google Colab, Python 3, NumPy, pandas, matplotlib, Gymnasium (ข้อ 6), Docker Desktop (ข้อ 7 เสริม) |
| เวลาที่ใช้ | 3 ชั่วโมง |
| สิ่งที่ต้องส่ง | ไฟล์ `.ipynb` ที่รันครบทุกเซลล์ พร้อมคำตอบคำถามท้ายข้อ (ดูหัวข้อ "สิ่งที่ต้องส่ง") |

### วัตถุประสงค์

1. คำนวณ discounted return และการอัปเดต Q-value ด้วยมือ แล้วตรวจสอบด้วยโค้ด
2. เขียนฟังก์ชัน ε-greedy และเข้าใจผลของการลดค่า ε
3. เขียน Q-Learning ให้ agent เรียนรู้การเล่นเกม Grid World แสดงกราฟ reward และ policy ที่ได้
4. ทดลองปรับ α, γ, ε แล้ววิเคราะห์ผล
5. ออกแบบ reward สังเกตพฤติกรรมของ agent และอธิบายปรากฏการณ์ reward hacking
6. นำโค้ดเดียวกันไปใช้กับ environment มาตรฐาน `FrozenLake-v1` ของ Gymnasium

> [!NOTE]
> ทบทวนเนื้อหาที่ [บทที่ 9](/module/09) ก่อนทำใบงาน โดยเฉพาะหัวข้อ 2 (return), 4 (Q-Learning, ε-greedy) และ 6 (reward design) ใบงานนี้ใช้ Grid World ขนาด **5×5** ซึ่งต่างจากตัวอย่าง 4×4 ในบทเรียน

## ข้อที่ 0: เตรียมสภาพแวดล้อม

### โจทย์

รันเซลล์ด้านล่างเพื่อ import ไลบรารี (ข้อ 1–5 ใช้แค่ NumPy, pandas และ matplotlib ส่วน Gymnasium จะติดตั้งในข้อ 6)

### โค้ดเริ่มต้น

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)
print("NumPy", np.__version__)
```

### ผลลัพธ์ที่คาดหวัง

```output
NumPy 2.x.x
```

(เลขเวอร์ชันขึ้นกับ Colab ขณะนั้น)

## ข้อที่ 1: Discounted Return และ Q-update ด้วยมือ

### โจทย์

**1.1** agent เล่นเกมหนึ่ง episode ได้ reward ตามลำดับ $r_1, \dots, r_5 = [0, 0, -1, 0, 5]$ จงคำนวณ $G_0 = \sum_k \gamma^k r_{k+1}$ **ด้วยมือ** (เขียนลงกระดาษหรือ markdown cell) เมื่อ $\gamma = 0.5$ และ $\gamma = 0.9$ แล้วเขียนฟังก์ชัน `discounted_return` เพื่อตรวจคำตอบ

**1.2** กำหนด $Q(s,a) = 2.0$, ได้ reward $r = 1$, state ถัดไปมี $Q(s', \cdot) = [1.5, 3.0]$, $\alpha = 0.5$, $\gamma = 0.9$ จงคำนวณ TD target, TD error และ $Q(s,a)$ ใหม่ ด้วยมือก่อน แล้วเติมโค้ดให้ได้ผลตรงกัน

$$ Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right] $$

### โค้ดเริ่มต้น

```python
def discounted_return(rewards, gamma):
    """G_0 = r_1 + γ r_2 + γ² r_3 + ... (rewards[0] คือ r_1)"""
    G = 0.0
    # TODO: วนลูป k, r ใน enumerate(rewards) แล้วบวก (gamma ** k) * r เข้าไปใน G
    return G

rewards = [0, 0, -1, 0, 5]
for gamma in [0.0, 0.5, 0.9, 1.0]:
    print(f"gamma = {gamma:.1f} -> G0 = {discounted_return(rewards, gamma):.4f}")
```

```python
alpha, gamma = 0.5, 0.9
q_sa   = 2.0                    # Q(s, a) เดิม
r      = 1.0                    # reward ที่ได้หลังทำ action a
q_next = np.array([1.5, 3.0])   # Q(s', ·) ของ state ถัดไป (มี 2 actions)

td_target = 0.0   # TODO: r + gamma * (ค่า Q สูงสุดของ state ถัดไป)
td_error  = 0.0   # TODO: td_target - q_sa
q_new     = q_sa  # TODO: q_sa + alpha * td_error

print(f"TD target   = {td_target:.4f}")
print(f"TD error    = {td_error:.4f}")
print(f"Q(s,a) ใหม่ = {q_new:.4f}")
```

### ผลลัพธ์ที่คาดหวัง

```output
gamma = 0.0 -> G0 = 0.0000
gamma = 0.5 -> G0 = 0.0625
gamma = 0.9 -> G0 = 2.4705
gamma = 1.0 -> G0 = 4.0000
```

```output
TD target   = 3.7000
TD error    = 1.7000
Q(s,a) ใหม่ = 2.8500
```

### คำถามท้ายข้อ

1. ทำไม $G_0$ เมื่อ $\gamma = 0.5$ จึงเกือบเป็นศูนย์ ทั้งที่มี reward +5 อยู่ในตอนท้าย
2. ถ้า $s'$ ในข้อ 1.2 เป็น terminal state (เกมจบ) ค่า $Q(s,a)$ ใหม่จะเป็นเท่าใด แสดงวิธีคิด
3. ถ้าเปลี่ยน $\alpha$ จาก 0.5 เป็น 1.0 ค่า $Q(s,a)$ ใหม่จะเท่ากับอะไร และการตั้ง $\alpha = 1$ มีความเสี่ยงอย่างไรใน environment ที่มีความสุ่ม

## ข้อที่ 2: กลยุทธ์ ε-greedy

### โจทย์

เขียนฟังก์ชัน `epsilon_greedy(q_row, epsilon, rng)` ที่

- ด้วยความน่าจะเป็น `epsilon` คืน action แบบสุ่ม (exploration)
- ไม่เช่นนั้นคืน action ที่ค่า Q สูงสุด (exploitation) **ถ้ามีหลาย action ที่ค่าเท่ากัน ให้สุ่มเลือกหนึ่งตัว** (เพราะตอนเริ่มต้น Q-table เป็น 0 ทั้งหมด)

จากนั้นทดสอบว่าเลือก action ที่ดีที่สุดบ่อยแค่ไหนเมื่อ ε ต่างกัน และดูกราฟการลดค่า ε (ε-decay)

### โค้ดเริ่มต้น

```python
def epsilon_greedy(q_row, epsilon, rng):
    """เลือก action จาก Q(s, ·) ด้วยกลยุทธ์ ε-greedy"""
    # TODO 1: ถ้า rng.random() < epsilon ให้ return action สุ่ม (ใช้ rng.integers(len(q_row)))
    # TODO 2: ไม่เช่นนั้นหา index ของ action ที่ Q สูงสุดทั้งหมด (ใช้ np.flatnonzero)
    #         แล้วสุ่มเลือกหนึ่งตัวด้วย rng.choice(...) และ return เป็น int
    return 0

rng = np.random.default_rng(0)
q_row = np.array([0.2, 0.5, 0.9, 0.1])
for eps in [0.0, 0.2, 1.0]:
    picks = np.array([epsilon_greedy(q_row, eps, rng) for _ in range(10_000)])
    print(f"epsilon = {eps:.1f}: เลือก action 2 (ดีที่สุด) {np.mean(picks == 2):.1%}")
```

```python
eps, eps_min, eps_decay = 1.0, 0.05, 0.995
schedule = []
for ep in range(1000):
    schedule.append(eps)
    eps = max(eps_min, eps * eps_decay)

plt.figure(figsize=(6, 3))
plt.plot(schedule)
plt.xlabel("episode"); plt.ylabel("epsilon"); plt.title("Epsilon decay")
plt.show()
print("epsilon ที่ episode 0, 100, 300, 600:", [round(schedule[i], 3) for i in (0, 100, 300, 600)])
```

### ผลลัพธ์ที่คาดหวัง

```output
epsilon = 0.0: เลือก action 2 (ดีที่สุด) 100.0%
epsilon = 0.2: เลือก action 2 (ดีที่สุด) 85.5%
epsilon = 1.0: เลือก action 2 (ดีที่สุด) 25.0%
```

```output
epsilon ที่ episode 0, 100, 300, 600: [1.0, 0.606, 0.222, 0.05]
```

กราฟ ε ลดลงแบบ exponential จาก 1.0 แล้วคงที่ที่ 0.05 ตั้งแต่ประมาณ episode 600

### คำถามท้ายข้อ

1. คำนวณค่าทางทฤษฎีของ "โอกาสเลือก action 2" เมื่อ ε = 0.2 (มี 4 actions) เทียบกับผลที่ได้
2. ถ้าใช้ `np.argmax(q_row)` ตรง ๆ โดยไม่สุ่มเมื่อค่าเสมอกัน ตอนที่ Q-table เป็น 0 ทั้งหมด agent จะทำอะไร
3. ทำไมจึงไม่ลด ε ให้เหลือ 0 พอดี แต่หยุดที่ค่าต่ำสุด (`eps_min`)

## ข้อที่ 3: Q-Learning บน Grid World

### โจทย์

เกม Grid World 5×5 (S = เริ่ม, G = เป้าหมาย +1, T = กับดัก −1, # = กำแพง, เดินหนึ่งก้าว −0.04, ชนขอบ/กำแพงอยู่ที่เดิม, ตัดจบที่ 100 ก้าว)

```
S . . # .        state ของช่อง (แถว r, คอลัมน์ c) = 5r + c
. # . . .        S = state 0, G = state 24
. . . # T        T อยู่ที่ state 14 และ 15
T # . . .
. . . # G
```

1. รันเซลล์สร้าง environment (ให้มาครบแล้ว) และอ่านให้เข้าใจว่า `step()` คืนค่าอะไร
2. เติม **กฎการอัปเดตของ Q-Learning** ใน `q_learning()` (TODO)
3. พล็อตกราฟ reward และจำนวนก้าวต่อ episode
4. แสดง policy เป็นลูกศร ตาราง $V(s) = \max_a Q(s,a)$ และเส้นทางเมื่อเล่นแบบ greedy

### โค้ดเริ่มต้น

```python
class GridWorld:
    """Grid World 5×5 ที่มี API แบบเดียวกับ Gymnasium

    S = จุดเริ่ม, G = เป้าหมาย, T = กับดัก, # = กำแพง
    actions: 0=ขึ้น 1=ขวา 2=ลง 3=ซ้าย  (ชนขอบ/กำแพง -> อยู่ที่เดิม)
    """
    MAP = ["S..#.",
           ".#...",
           "...#T",
           "T#...",
           "...#G"]
    ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    ARROWS = "↑→↓←"

    def __init__(self, step_reward=-0.04, goal_reward=1.0, trap_reward=-1.0, max_steps=100):
        self.n_rows, self.n_cols = len(self.MAP), len(self.MAP[0])
        self.n_states, self.n_actions = self.n_rows * self.n_cols, 4
        self.step_reward, self.goal_reward, self.trap_reward = step_reward, goal_reward, trap_reward
        self.max_steps = max_steps

    def reset(self, seed=None):
        self.pos, self.t = (0, 0), 0
        return self.to_state(self.pos), {}

    def to_state(self, pos):
        return pos[0] * self.n_cols + pos[1]

    def move(self, pos, action):
        dr, dc = self.ACTIONS[action]
        r, c = pos[0] + dr, pos[1] + dc
        if 0 <= r < self.n_rows and 0 <= c < self.n_cols and self.MAP[r][c] != "#":
            return (r, c)
        return pos

    def reward_of(self, old_pos, new_pos):
        cell = self.MAP[new_pos[0]][new_pos[1]]
        if cell == "G":
            return self.goal_reward, True
        if cell == "T":
            return self.trap_reward, True
        return self.step_reward, False

    def step(self, action):
        old_pos = self.pos
        self.pos = self.move(self.pos, action)
        self.t += 1
        reward, terminated = self.reward_of(old_pos, self.pos)
        truncated = (not terminated) and self.t >= self.max_steps
        return self.to_state(self.pos), reward, terminated, truncated, {}


def show_policy(env, Q):
    """พิมพ์ policy เป็นลูกศร (greedy: argmax Q)"""
    for r in range(env.n_rows):
        line = ""
        for c in range(env.n_cols):
            cell = env.MAP[r][c]
            line += f" {cell} " if cell in "#GT" else f" {env.ARROWS[int(np.argmax(Q[r * env.n_cols + c]))]} "
        print(line)


env = GridWorld()
s, _ = env.reset()
print("state เริ่มต้น =", s, "| จำนวน states =", env.n_states, "| จำนวน actions =", env.n_actions)
print("เดินลง ->", env.step(2))
```

```python
def get_sizes(env):
    """รองรับทั้ง GridWorld ของเรา และ environment ของ Gymnasium"""
    if hasattr(env, "observation_space"):
        return env.observation_space.n, env.action_space.n
    return env.n_states, env.n_actions


def q_learning(env, n_episodes=800, alpha=0.1, gamma=0.9,
               eps_start=1.0, eps_min=0.05, eps_decay=0.995, seed=42):
    rng = np.random.default_rng(seed)
    n_states, n_actions = get_sizes(env)
    Q = np.zeros((n_states, n_actions))
    rewards, lengths = [], []
    epsilon = eps_start
    for ep in range(n_episodes):
        s, _ = env.reset()
        total, steps, done = 0.0, 0, False
        while not done:
            a = epsilon_greedy(Q[s], epsilon, rng)
            s_next, r, terminated, truncated, _ = env.step(a)
            # TODO 1: TD target = r ถ้า terminated (ไม่มีอนาคต) ไม่เช่นนั้น r + gamma * max Q(s_next, ·)
            target = 0.0
            # TODO 2: อัปเดต Q[s, a] ด้วย alpha * (target - Q[s, a])
            Q[s, a] += 0.0
            s, total, steps = s_next, total + r, steps + 1
            done = terminated or truncated
        epsilon = max(eps_min, epsilon * eps_decay)
        rewards.append(total)
        lengths.append(steps)
    return Q, np.array(rewards), np.array(lengths)


Q, rewards, lengths = q_learning(env)
print("reward เฉลี่ย 100 episodes แรก    :", round(rewards[:100].mean(), 3))
print("reward เฉลี่ย 100 episodes สุดท้าย:", round(rewards[-100:].mean(), 3))
print("จำนวนก้าวเฉลี่ย 100 episodes สุดท้าย:", round(lengths[-100:].mean(), 1))
```

```python
def moving_average(x, w=50):
    return np.convolve(x, np.ones(w) / w, mode="valid")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].plot(rewards, alpha=0.3, label="reward per episode")
axes[0].plot(np.arange(49, len(rewards)), moving_average(rewards), label="moving average 50")
axes[0].set_xlabel("episode"); axes[0].set_ylabel("total reward"); axes[0].legend()
axes[1].plot(lengths, alpha=0.3)
axes[1].plot(np.arange(49, len(lengths)), moving_average(lengths))
axes[1].set_xlabel("episode"); axes[1].set_ylabel("steps per episode")
plt.tight_layout()
plt.show()
```

```python
def run_greedy(env, Q, max_steps=100):
    s, _ = env.reset()
    path, total = [s], 0.0
    for _ in range(max_steps):
        s, r, terminated, truncated, _ = env.step(int(np.argmax(Q[s])))
        path.append(s)
        total += r
        if terminated or truncated:
            break
    return path, total

show_policy(env, Q)
print("\nV(s) = max_a Q(s, a)")
print(Q.max(axis=1).reshape(env.n_rows, env.n_cols))
path, total = run_greedy(env, Q)
print("\nเส้นทาง greedy:", path)
print(f"จำนวนก้าว = {len(path) - 1}, reward รวม = {total:.2f}")
```

### ผลลัพธ์ที่คาดหวัง

```output
state เริ่มต้น = 0 | จำนวน states = 25 | จำนวน actions = 4
เดินลง -> (5, -0.04, False, False, {})
```

```output
reward เฉลี่ย 100 episodes แรก    : -1.564
reward เฉลี่ย 100 episodes สุดท้าย: 0.646
จำนวนก้าวเฉลี่ย 100 episodes สุดท้าย: 8.4
```

กราฟซ้าย: reward ช่วงแรกติดลบ (ตกกับดักหรือเดินวนจนหมด 100 ก้าว ได้ราว −4) แล้วเพิ่มขึ้นจนค่าเฉลี่ยเคลื่อนที่อยู่ราว 0.6–0.7 กราฟขวา: จำนวนก้าวลดจากหลักสิบลงมาใกล้ 8

```output
 →  →  ↓  #  → 
 ↓  #  ↓  ←  ← 
 →  →  ↓  #  T 
 T  #  →  →  ↓ 
 ←  →  ↑  #  G 

V(s) = max_a Q(s, a)
[[ 0.27   0.344  0.427  0.    -0.082]
 [ 0.186  0.     0.519  0.405 -0.079]
 [ 0.355  0.517  0.621  0.     0.   ]
 [ 0.     0.     0.734  0.86   1.   ]
 [-0.016  0.136  0.614  0.     0.   ]]

เส้นทาง greedy: [0, 1, 2, 7, 12, 17, 18, 19, 24]
จำนวนก้าว = 8, reward รวม = 0.72
```

### คำถามท้ายข้อ

1. ตรวจสอบ $V(\text{start}) = 0.27$ ด้วยมือ: เส้นทางสั้นที่สุดมี 8 ก้าว (7 ก้าวได้ −0.04 แล้วก้าวที่ 8 ได้ +1) ใช้ $\gamma = 0.9$ ได้ $G_0$ เท่าใด
2. ทำไมค่า $V$ ของช่อง state 23 (กำแพง) และช่องกับดัก/เป้าหมายจึงเป็น 0
3. ช่อง state 4 (มุมขวาบน) ทางสั้นที่สุดไป G ยาวกี่ก้าว ค่า $V$ ที่ถูกต้องควรเป็นเท่าใด ทำไมค่าที่เรียนได้ (−0.082) และลูกศร (→ ชนขอบ) จึงยังไม่ถูกต้อง (ใบ้: agent เดินผ่านมุมนี้บ่อยแค่ไหน)
4. reward เฉลี่ย 100 episodes สุดท้าย (0.646) ต่ำกว่า reward ของเส้นทาง greedy (0.72) เพราะอะไร

## ข้อที่ 4: ทดลองปรับ α, γ และ ε

### โจทย์

รันการทดลองด้านล่าง ซึ่งฝึก 5 seeds ต่อการตั้งค่าแล้วเฉลี่ยผล ตัวชี้วัดคือ

- `reward_last100` reward เฉลี่ยระหว่างฝึก 100 episodes สุดท้าย
- `episodes_to_0.5` episode แรกที่ค่าเฉลี่ยเคลื่อนที่ 50 episodes ≥ 0.5 (ยิ่งน้อยยิ่งเรียนเร็ว, NaN = ไม่ถึง)
- `V(start)` ค่าที่ agent ประเมินให้ state เริ่มต้น
- `greedy_steps`, `success_rate` จำนวนก้าวและอัตราการถึง G เมื่อเล่นแบบ greedy หลังฝึก

**เพิ่มการทดลองของตัวเองอย่างน้อย 1 แบบ** (TODO) แล้วสรุปผลเป็นตารางและข้อความ

### โค้ดเริ่มต้น

```python
def evaluate_config(alpha=0.1, gamma=0.9, eps_start=1.0, eps_decay=0.995, seeds=(0, 1, 2, 3, 4)):
    """ฝึกหลาย seed แล้วรายงานค่าเฉลี่ย"""
    last100, learn_ep, v_start, steps, success = [], [], [], [], []
    for seed in seeds:
        env_x = GridWorld()
        Q_x, rew_x, _ = q_learning(env_x, alpha=alpha, gamma=gamma,
                                   eps_start=eps_start, eps_decay=eps_decay, seed=seed)
        path, total = run_greedy(env_x, Q_x)
        v_start.append(Q_x[0].max())
        last100.append(rew_x[-100:].mean())
        ma = moving_average(rew_x, 50)
        hit = np.flatnonzero(ma >= 0.5)
        learn_ep.append(hit[0] + 49 if len(hit) else np.nan)   # episode แรกที่ค่าเฉลี่ย 50 ep ≥ 0.5
        steps.append(len(path) - 1)
        success.append(path[-1] == 24)          # state 24 = G
    return {"reward_last100": np.mean(last100), "episodes_to_0.5": np.mean(learn_ep),
            "V(start)": np.mean(v_start), "greedy_steps": np.mean(steps),
            "success_rate": np.mean(success)}

configs = {
    "baseline (a=0.1, g=0.9)": dict(),
    "alpha = 0.01":            dict(alpha=0.01),
    "alpha = 0.9":             dict(alpha=0.9),
    "gamma = 0.5":             dict(gamma=0.5),
    "gamma = 0.99":            dict(gamma=0.99),
    "epsilon = 0 (no explore)": dict(eps_start=0.0),
    "epsilon = 1 (random)":    dict(eps_decay=1.0),
    # TODO: เพิ่มการทดลองของตัวเองอย่างน้อย 1 แบบ เช่น "eps_decay = 0.95": dict(eps_decay=0.95),
}
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", None)
results = pd.DataFrame({name: evaluate_config(**cfg) for name, cfg in configs.items()}).T
print(results.round(3))
```

### ผลลัพธ์ที่คาดหวัง

(ไม่รวมแถวที่นักศึกษาเพิ่มเอง)

```output
                          reward_last100  episodes_to_0.5  V(start)  greedy_steps  success_rate
baseline (a=0.1, g=0.9)            0.684            338.2     0.270           8.0           1.0
alpha = 0.01                       0.675            420.4    -0.060           8.0           1.0
alpha = 0.9                        0.673            308.6     0.270           8.0           1.0
gamma = 0.5                        0.682            349.0    -0.072           8.0           1.0
gamma = 0.99                       0.670            334.8     0.660           8.0           1.0
epsilon = 0 (no explore)           0.653             65.8     0.270           8.0           1.0
epsilon = 1 (random)              -2.406              NaN     0.250           8.0           1.0
```

### คำถามท้ายข้อ

1. `alpha = 0.01` ได้ `V(start)` = −0.060 ห่างจากค่าจริง 0.270 มาก แต่ policy ยังถูกต้อง (8 ก้าว) อธิบายว่าทำไม
2. คำนวณด้วยมือว่าค่าที่ถูกต้องของ $V(\text{start})$ เมื่อ $\gamma = 0.5$ และ $\gamma = 0.99$ คือเท่าใด (ใช้วิธีเดียวกับคำถามท้ายข้อ 3 ข้อ 1) เทียบกับตาราง ค่า $\gamma$ มีผลต่อ "ขนาด" ของค่า Q อย่างไร
3. `epsilon = 1 (random)` เดินสุ่มตลอดการฝึก ได้ reward ระหว่างฝึกแย่มาก แต่ทำไม policy แบบ greedy หลังฝึกยังไปถึง G ได้ 100% (ใบ้: Q-Learning เป็น off-policy)
4. `epsilon = 0` เรียนรู้ได้เร็วที่สุดในเกมนี้ ทั้งที่ไม่สุ่มเลย อธิบายโดยพิจารณาว่า Q เริ่มต้นเป็น 0 แต่ reward ต่อก้าวเป็น −0.04 (ใบ้: action ที่ลองแล้วจะมีค่าติดลบ ส่วน action ที่ยังไม่ลองยังเป็น 0) แล้ววิเคราะห์ว่าใน FrozenLake ที่ reward ต่อก้าวเป็น 0 กลไกนี้ยังได้ผลหรือไม่
5. สรุปผลการทดลองที่นักศึกษาเพิ่มเอง

## ข้อที่ 5: ออกแบบ Reward และ Reward Hacking

### โจทย์

**5.1** รันการทดลองเปลี่ยน reward ต่อก้าว (`step_reward`) 4 แบบ แล้วบันทึกพฤติกรรมของ agent

**5.2** ออกแบบชุด reward ของตัวเอง (TODO) เช่น ทำให้ agent "กลัว" กับดักมากขึ้น หรือรีบไปถึง G มากขึ้น แล้วอธิบายผล

**5.3** เปรียบเทียบ reward shaping สองแบบ ที่ให้โบนัสเมื่อเดินเข้าใกล้ G

- **closer bonus** ได้ +0.5 ทุกครั้งที่ระยะ Manhattan ถึง G ลดลง แต่ไม่ถูกหักเมื่อถอยออก
- **potential-based** ให้ $F = \gamma\Phi(s') - \Phi(s)$ โดย $\Phi(s) = -0.5 \times$ ระยะถึง G

### โค้ดเริ่มต้น

```python
for step_reward in [-0.04, 0.0, -1.0, +0.2]:
    env_r = GridWorld(step_reward=step_reward)
    Q_r, rew_r, _ = q_learning(env_r, n_episodes=1500)
    path, total = run_greedy(env_r, Q_r)
    end = {24: "G", 14: "T", 15: "T"}.get(path[-1], f"state {path[-1]}")
    print(f"step_reward = {step_reward:+.2f}: เดิน {len(path) - 1:3d} ก้าว จบที่ {end}, reward รวม = {total:.2f}")
    show_policy(env_r, Q_r)
    print()
```

```python
# TODO: ออกแบบ reward ของตัวเอง แล้วอธิบายผลใน markdown cell ถัดไป
my_rewards = dict(step_reward=-0.04, goal_reward=1.0, trap_reward=-1.0)

env_my = GridWorld(**my_rewards)
Q_my, rew_my, _ = q_learning(env_my, n_episodes=1500)
path, total = run_greedy(env_my, Q_my)
print(my_rewards, "-> เดิน", len(path) - 1, "ก้าว จบที่ state", path[-1])
show_policy(env_my, Q_my)
```

```python
class CloserBonusGridWorld(GridWorld):
    """reward shaping แบบไร้เดียงสา: ได้โบนัสเมื่อเข้าใกล้ G (ระยะ Manhattan ลดลง) แต่ไม่ถูกหักเมื่อถอยออก"""
    GOAL = (4, 4)

    def __init__(self, bonus=0.5, **kw):
        super().__init__(**kw)
        self.bonus = bonus

    def distance(self, pos):
        return abs(pos[0] - self.GOAL[0]) + abs(pos[1] - self.GOAL[1])

    def reward_of(self, old_pos, new_pos):
        reward, terminated = super().reward_of(old_pos, new_pos)
        if self.distance(new_pos) < self.distance(old_pos):
            reward += self.bonus
        return reward, terminated


class PotentialShapedGridWorld(CloserBonusGridWorld):
    """potential-based shaping: F = γ·Φ(s') − Φ(s) โดย Φ(s) = −bonus × ระยะถึง G"""
    def __init__(self, gamma=0.9, **kw):
        super().__init__(**kw)
        self.gamma = gamma

    def reward_of(self, old_pos, new_pos):
        reward, terminated = GridWorld.reward_of(self, old_pos, new_pos)
        phi_old = -self.bonus * self.distance(old_pos)
        phi_new = 0.0 if terminated else -self.bonus * self.distance(new_pos)
        return reward + self.gamma * phi_new - phi_old, terminated


for name, env_s in [("closer bonus", CloserBonusGridWorld()),
                    ("potential-based", PotentialShapedGridWorld())]:
    Q_s, rew_s, _ = q_learning(env_s, n_episodes=1500)
    path, total = run_greedy(env_s, Q_s)
    print(f"{name}: เดิน {len(path) - 1} ก้าว, จบที่ state {path[-1]}, reward (หลัง shaping) รวม = {total:.2f}")
    show_policy(env_s, Q_s)
    print()
```

### ผลลัพธ์ที่คาดหวัง

```output
step_reward = -0.04: เดิน   8 ก้าว จบที่ G, reward รวม = 0.72
 →  →  ↓  #  → 
 ↓  #  ↓  ←  ← 
 →  →  ↓  #  T 
 T  #  →  →  ↓ 
 ←  →  ↑  #  G 

step_reward = +0.00: เดิน   8 ก้าว จบที่ G, reward รวม = 1.00
 →  →  ↓  #  ↓ 
 ↑  #  ↓  ←  ← 
 →  →  ↓  #  T 
 T  #  →  →  ↓ 
 →  →  ↑  #  G 

step_reward = -1.00: เดิน   3 ก้าว จบที่ T, reward รวม = -3.00
 ↓  ←  ↓  #  ↓ 
 ↓  #  →  →  ↓ 
 ↓  ←  ↓  #  T 
 T  #  →  →  ↓ 
 ↓  ↓  ←  #  G 

step_reward = +0.20: เดิน 100 ก้าว จบที่ state 0, reward รวม = 20.00
 ↑  ↑  ←  #  → 
 ↑  #  ↑  ←  ↑ 
 ↑  ←  ↑  #  T 
 T  #  ↑  ←  ↓ 
 →  ↑  ↑  #  G 
```

ข้อ 5.2 ผลขึ้นกับ reward ที่นักศึกษาออกแบบ

```output
closer bonus: เดิน 100 ก้าว, จบที่ state 18, reward (หลัง shaping) รวม = 22.50
 →  →  ↓  #  ↓ 
 ↑  #  ↓  ←  ← 
 →  →  ↓  #  T 
 T  #  →  →  ← 
 →  →  ↑  #  G 

potential-based: เดิน 8 ก้าว, จบที่ state 24, reward (หลัง shaping) รวม = 6.12
 →  →  ↓  #  ↓ 
 ↑  #  ↓  ←  ↓ 
 ↓  →  ↓  #  T 
 T  #  →  →  ↓ 
 →  →  ↑  #  G 
```

### คำถามท้ายข้อ

1. เมื่อ `step_reward = -1.0` agent เลือกเดินลง 3 ก้าวไปตกกับดัก T ที่ state 15 คำนวณ discounted return ($\gamma = 0.9$) ของ "ทางไปกับดัก" (3 ก้าว) เทียบกับ "ทางไป G" (8 ก้าว) เพื่อแสดงว่า agent ตัดสินใจอย่าง "มีเหตุผล" ตาม reward ที่ได้รับ
2. เมื่อ `step_reward = +0.2` agent ไม่ยอมไปถึง G อธิบายด้วยค่า $0.2/(1-\gamma)$ และบอกว่าพฤติกรรมนี้คือ reward hacking อย่างไร
3. `step_reward = 0.0` agent ยังไปถึง G ด้วยเส้นทางสั้นที่สุด ทั้งที่การเดินไม่มีต้นทุน อะไรทำให้ agent ยัง "รีบ"
4. ใน closer bonus ดูลูกศรที่ state 18 และ 19 (แถว 3 คอลัมน์ 3–4) agent hack reward อย่างไร ทำไม potential-based shaping จึงไม่เกิดปัญหานี้
5. อธิบายชุด reward ที่ออกแบบเองในข้อ 5.2 และผลที่ได้ ยกตัวอย่าง reward hacking ในเกมหรือระบบจริงอีก 1 ตัวอย่าง

## ข้อที่ 6: FrozenLake-v1 ด้วย Gymnasium

### โจทย์

`FrozenLake-v1` เป็นเกมเดินบนทะเลสาบน้ำแข็ง 4×4 (S เริ่ม, F พื้นน้ำแข็ง, H หลุม = จบเกม, G เป้าหมาย = +1) reward เป็น 0 ทุกก้าวยกเว้นตอนถึง G (sparse reward) action คือ 0=ซ้าย 1=ลง 2=ขวา 3=ขึ้น

```
S F F F
F H F H
F F F H
H F F G
```

1. ติดตั้ง/import Gymnasium แล้วสำรวจ `observation_space`, `action_space`, `reset()` และ `step()`
2. ใช้ฟังก์ชัน `q_learning()` **ตัวเดิมจากข้อ 3** ฝึกบน FrozenLake แบบไม่ลื่น (`is_slippery=False`) และแบบลื่น (`is_slippery=True` ไปทิศที่ตั้งใจเพียง 1/3)
3. วัด success rate จากการเล่นแบบ greedy 1000 episodes และพล็อตอัตราสำเร็จระหว่างฝึก

### โค้ดเริ่มต้น

```python
try:
    import gymnasium as gym
except ImportError:                        # ติดตั้งเมื่อยังไม่มี
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gymnasium"], check=True)
    import gymnasium as gym

env_fl = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
print("observation_space:", env_fl.observation_space, "| action_space:", env_fl.action_space)
obs, info = env_fl.reset(seed=42)
print("state เริ่มต้น:", obs)
print("step(ขวา):", env_fl.step(2))         # FrozenLake: 0=ซ้าย 1=ลง 2=ขวา 3=ขึ้น
```

```python
def success_rate(env, Q, n_episodes=1000):
    wins = 0
    for _ in range(n_episodes):
        s, _ = env.reset()
        done = False
        while not done:
            s, r, terminated, truncated, _ = env.step(int(np.argmax(Q[s])))
            done = terminated or truncated
        wins += r > 0
    return wins / n_episodes

# 6.1 พื้นไม่ลื่น (deterministic)
env_fl = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
env_fl.reset(seed=42)
Q_fl, rew_fl, _ = q_learning(env_fl, n_episodes=1000, alpha=0.1, gamma=0.95, eps_decay=0.995)
print("ไม่ลื่น : success rate =", success_rate(env_fl, Q_fl))

# 6.2 พื้นลื่น (stochastic) — ต้องใช้ episode มากขึ้น และ γ ใกล้ 1
env_sl = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
env_sl.reset(seed=42)
Q_sl, rew_sl, _ = q_learning(env_sl, n_episodes=10000, alpha=0.05, gamma=0.99, eps_decay=0.9995)
print("ลื่น    : success rate =", success_rate(env_sl, Q_sl))

arrows = "←↓→↑"                               # ลำดับ action ของ FrozenLake
lake = ["SFFF", "FHFH", "FFFH", "HFFG"]
for r in range(4):
    print(" ".join(lake[r][c] if lake[r][c] in "HG" else arrows[int(np.argmax(Q_sl[r * 4 + c]))]
                   for c in range(4)))
```

```python
plt.figure(figsize=(7, 3))
plt.plot(np.arange(99, len(rew_sl)), moving_average(rew_sl, 100))
plt.xlabel("episode"); plt.ylabel("success rate (moving avg 100)")
plt.title("FrozenLake-v1 (slippery) during training")
plt.show()
```

### ผลลัพธ์ที่คาดหวัง

ตัวอย่างผลลัพธ์ (ค่าที่ได้จะต่างกันเล็กน้อยตามเวอร์ชันของ Gymnasium และการสุ่ม)

```output
observation_space: Discrete(16) | action_space: Discrete(4)
state เริ่มต้น: 0
step(ขวา): (1, 0.0, False, False, {'prob': 1.0})
```

- แบบไม่ลื่น: success rate = 1.0 (เดินทางสั้นที่สุด 6 ก้าวได้ทุกครั้ง)
- แบบลื่น: success rate ประมาณ 0.6–0.75 (ผู้สอนทดลองจำลองกฎเดียวกันได้ราว 0.7) ไม่ถึง 1.0 แม้ policy จะดีที่สุดแล้ว เพราะการไถลทำให้ตกหลุมได้เสมอ policy ที่ได้มักมีลักษณะ "เดินเลียบขอบและหันหลังให้หลุม" เช่น

```output
← ↑ ← ←
← H ← H
↑ ↓ ← H
H → ↓ G
```

กราฟอัตราสำเร็จระหว่างฝึกจะเริ่มใกล้ 0 (สุ่มเกือบล้วน) แล้วค่อย ๆ เพิ่มขึ้นเมื่อ ε ลดลง

### คำถามท้ายข้อ

1. ทำไมฟังก์ชัน `q_learning()` จากข้อ 3 จึงใช้กับ FrozenLake ได้โดยไม่ต้องแก้ (พิจารณา API ของ `reset()` และ `step()`)
2. FrozenLake ให้ reward แบบ sparse ถ้า `epsilon_greedy` ไม่สุ่มเมื่อค่า Q เสมอกัน จะเกิดอะไรขึ้น (ทดลองได้โดยแก้ชั่วคราวเป็น `np.argmax`)
3. ทำไมแบบลื่นจึงต้องใช้ episode มากกว่า, α เล็กกว่า และ γ ใกล้ 1 มากกว่าแบบไม่ลื่น
4. ที่ช่องแถว 1 คอลัมน์ 0 (ใต้ S) policy มักเลือก "ซ้าย" ซึ่งดูเหมือนชนขอบ อธิบายว่าทำไมจึงเป็นทางเลือกที่ปลอดภัยเมื่อพื้นลื่น
5. (ท้าทาย) เปลี่ยนเป็น `map_name="8x8"` แล้วรายงานว่าต้องปรับ hyperparameter อะไรบ้าง

## ข้อที่ 7 (เสริม): เว็บแล็บ RL ของรายวิชา

### โจทย์

ผู้สอนเตรียมเว็บแอปจำลอง RL ไว้ในโฟลเดอร์ `02 Machine Learning/Machine Learning Vault/04_Reinforcement_Learning/cartpole_web` ข้อนี้รันบนเครื่องของตนเองที่มี Docker Desktop (ไม่ใช่ Colab) หรือดูการสาธิตในชั้นเรียน

```bash
cd "02 Machine Learning/Machine Learning Vault/04_Reinforcement_Learning/cartpole_web"
docker compose up -d --build
# เปิด browser ที่ http://localhost:8000  และหยุดด้วย  docker compose down
```

1. หน้า `/cartpole` เปรียบเทียบ agent ทั้ง 4 แบบ (Random, Heuristic, Q-Learning pretrained, Q-Learning from scratch + Fast-train)
2. หน้า `/qtable` เริ่มจาก Q-table ว่าง กด Step ทีละครั้งอย่างน้อย 10 ครั้ง สังเกตการ discretize, ε-greedy และการแทนค่าสมการ Bellman จากนั้นกด "ฝึกเร็ว"
3. หน้า `/maze` ดู replay buffer, target network และกราฟจำนวนก้าวเทียบกับทางสั้นที่สุด

### คำถามท้ายข้อ

1. CartPole ใน Q-Table Lab แบ่ง state เป็นกี่ bin ในแต่ละมิติ (x, ẋ, θ, θ̇) และ Q-table มีทั้งหมดกี่ค่า ทำไมจึงให้ bin กับ θ และ θ̇ มากกว่า
2. จดตัวเลขจากการ update 1 step ในหน้า `/qtable` (r, γ, max Q(s′,·), Q(s,a) เดิม, α) แล้วคำนวณ Q ใหม่ด้วยมือ ตรงกับที่หน้าเว็บแสดงหรือไม่
3. ค่า Q ของ agent แบบ pretrained อยู่ราว 90 สอดคล้องกับ $1/(1-\gamma)$ อย่างไร
4. ใน Maze DQN ถ้า network ทำได้ดี ทำไม README จึงบอกว่า maze ขนาดเล็กใช้ Q-table ก็พอ และข้อได้เปรียบจริงของ DQN อยู่ที่ใด

## เกณฑ์การให้คะแนน

| ประเด็นการประเมิน | คะแนน | สิ่งที่ตรวจในใบงานนี้ |
|---|---|---|
| อธิบายแนวคิด นิยาม และหลักการได้ถูกต้อง | 2 | คำตอบเรื่อง return, γ, ε-greedy, off-policy, terminated/truncated (ข้อ 1, 2, 4) |
| วิเคราะห์ปัญหาและเลือกเทคนิคหรืออัลกอริทึมได้เหมาะสม | 2 | การวิเคราะห์ผลการทดลอง α/γ/ε และการเลือก hyperparameter ของ FrozenLake (ข้อ 4, 6) |
| แสดงขั้นตอน วิธีคิด pipeline หรือกระบวนการพัฒนาโมเดลได้ครบถ้วน | 2 | การคำนวณด้วยมือ (ข้อ 1, คำถามข้อ 3 และ 5) และ training loop ที่ครบ เลือก action → step → update → ลด ε |
| เขียนโค้ด สูตร การคำนวณ หรือการประเมินผลได้ถูกต้อง | 3 | `discounted_return`, `epsilon_greedy`, กฎ Q-Learning ทำงานถูกต้อง ผลตรงกับที่คาดหวัง กราฟและ policy ครบ |
| สรุปผล ยกตัวอย่าง ตีความผลลัพธ์ หรืออธิบายข้อจำกัดได้เหมาะสม | 1 | การอธิบาย reward hacking และการออกแบบ reward ของตนเอง (ข้อ 5) |
| **รวม** | **10** | ข้อ 7 เป็นข้อเสริม |

## สิ่งที่ต้องส่ง

1. ไฟล์ `Lab09_<รหัสนักศึกษา>.ipynb` ที่รันครบทุกเซลล์ตั้งแต่ข้อ 0–6 (Runtime → Run all) และเห็นผลลัพธ์/กราฟทุกเซลล์
2. คำตอบคำถามท้ายข้อทุกข้อ เขียนใน markdown cell ใต้เซลล์โค้ดของข้อนั้น (การคำนวณด้วยมือให้แสดงวิธีทำ)
3. ตารางผลการทดลองข้อ 4 ที่มีการทดลองของตนเองอย่างน้อย 1 แถว และชุด reward ที่ออกแบบเองในข้อ 5.2 พร้อมคำอธิบาย
4. (เสริม) ภาพหน้าจอจากเว็บแล็บข้อ 7 พร้อมคำตอบ
