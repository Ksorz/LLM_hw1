# Обучение языковой модели

## Эксперименты с гиперпараметрами

### Batch Size и Gradient Accumulation
- Проведено 15 экспериментов через SWEEP_CONFIG
- Лучший результат показало сочетание `per_device_train_batch_size=6` и `gradient_accumulation_steps=6`
- Возможно, большие значения дадут ещё больший буст

### Learning Rate и Scheduler
- При высоком lr (1e-4 или 5e-5) модель перестаёт стабильно сходиться при достижении loss ≈ 4
- Добавлен кастомный scheduler, который снижает lr после N шагов
- Изначально использовался triangular scheduler, но он показался избыточным
- Оставлено только снижение lr до определённого уровня

### Текущие результаты
- Снижение lr дало буст в обучении (результаты видны в wandb)
- Попытка снизить lr до ~5e-5 на 20-й минуте была не самой удачной
- Снижение lr помогает, но баланс сложно нащупать

### Планы на будущее
- Увеличить batch size и lr
- Сделать снижение lr более плавным
- Следующий эксперимент: `per_device_train_batch_size=10` и `gradient_accumulation_steps=8`

## Мониторинг
[Wandb Dashboard](https://wandb.ai/ksorzz/llm_hw1-aylesnov/workspace?nw=nwuserksorzz)