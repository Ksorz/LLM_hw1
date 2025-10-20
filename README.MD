# Обучение языковой модели

## Эксперименты с гиперпараметрами

### Batch Size и Gradient Accumulation
- Проведено 15 экспериментов через `wandb.sweep`
- Лучший результат показало сочетание `per_device_train_batch_size=6` и `gradient_accumulation_steps=6`
- Возможно, большие значения дадут ещё больший буст

### Learning Rate и Scheduler
- После 15 экспериментов через `wandb.sweep` я делал кастомные эксперименты руками
- При высоком lr (1e-4 или 5e-5) модель перестаёт стабильно сходиться при достижении loss ≈ 4
- Добавлен кастомный scheduler, который снижает lr после N шагов
- Изначально использовался triangular scheduler, но он показался избыточным (хотя и дал лучший результат)
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


## Примеры генераций на промпт "В начале было слово":
*Ограничение - 64 токена*

___
В начале было слово, и слово было сделано в СССР и в Москве.

В культуре 

В романе М. И. Степанова принимал участие в съёмках фильма «Вторая история» (1957), и в «Пушкинском театре» (1957).

Примечания

Ссылки 

Дети кино
wandb:                                                                                
wandb: 
wandb: Run history:
wandb:               eval/loss █▁▁
wandb:            eval/runtime ▃█▁
wandb: eval/samples_per_second ▆▁█
wandb:   eval/steps_per_second ▆▁█
wandb:             train/epoch ▁▁▂▂▂▂▂▂▃▃▃▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇▇███
wandb:       train/global_step ▁▂▂▂▂▂▂▂▂▃▃▃▃▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▆▆▇▇▇▇▇▇████
wandb:         train/grad_norm ▄█▃▂▄▃▂▃▂▂▂▁▁▁▃▂▃▂▁▂▃▂▂▁▁▃▄▁▁▂▂▂▂▃▁▂▁▁▂▂
wandb:     train/learning_rate ████▇▇▆▆▆▆▆▆▅▅▅▄▄▄▄▄▃▃▃▃▃▃▃▃▃▃▃▃▂▂▂▁▁▁▁▁
wandb:              train/loss █▇▅▅▄▄▄▃▃▃▃▃▂▂▃▂▂▂▂▂▂▂▂▂▁▂▂▁▁▂▂▁▁▁▁▁▁▂▁▁
wandb: 
wandb: Run summary:
wandb:                eval/loss 4.69116
wandb:             eval/runtime 47.6211
wandb:  eval/samples_per_second 104.996
wandb:    eval/steps_per_second 26.249
wandb:               total_flos 1.3433323670785229e+17
wandb:              train/epoch 0.02627
wandb:        train/global_step 6371
wandb:          train/grad_norm 1.19531
wandb:      train/learning_rate 0.00049
wandb:               train/loss 4.0917
wandb:               train_loss 4.68242
wandb:            train_runtime 1800.7458
wandb: train_samples_per_second 1077.366
wandb:   train_steps_per_second 134.671
___
В начале было слово, и слово было сделано от этого лица. В начале XIX века в XVIII веке на месте дома помещицы, в частности, «Афики» В. И. Гурова был убит под угрозой исследования в «Пушкинском государстве».

Примечания

Литература 

 Могии А
wandb:                                                                                
wandb: 
wandb: Run history:
wandb:               eval/loss █▁▁
wandb:            eval/runtime ▄▁█
wandb: eval/samples_per_second ▅█▁
wandb:   eval/steps_per_second ▅█▁
wandb:             train/epoch ▁▁▁▁▁▁▂▂▂▂▂▂▂▂▃▃▃▃▃▄▄▄▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇██
wandb:       train/global_step ▁▁▁▂▂▂▂▂▂▂▂▃▃▃▃▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇█████
wandb:         train/grad_norm █▅█▄▃▄▃▄▇▂▄▃▂▃▂▃▂▂▂▁▂▃▂▃▃▁▂▃▃▂▁▂▁▂▁▂▁▂▁▂
wandb:     train/learning_rate ██████████▇▇▇▇▆▆▆▆▆▆▆▆▅▅▅▄▄▄▄▃▃▃▃▃▃▂▂▂▁▁
wandb:              train/loss █▆▇▇▆▅▅▅▅▃▅▃▃▄▃▂▃▃▂▂▃▃▂▂▃▂▂▂▃▂▂▃▂▂▂▁▁▁▂▂
wandb: 
wandb: Run summary:
wandb:                eval/loss 4.71272
wandb:             eval/runtime 48.252
wandb:  eval/samples_per_second 103.623
wandb:    eval/steps_per_second 25.906
wandb:               total_flos 1.3500796023236198e+17
wandb:              train/epoch 0.0264
wandb:        train/global_step 6403
wandb:          train/grad_norm 1.1875
wandb:      train/learning_rate 0.0001
wandb:               train/loss 3.6472
wandb:               train_loss 4.72544
wandb:            train_runtime 1800.7671
wandb: train_samples_per_second 1077.354
wandb:   train_steps_per_second 134.669
___
В начале было слово, и слово было дано в честь святого Иоанна Иоанна Предтеи Ибн.

В честь святого Иоанна Дамасканцев, которое в честь святого Иоанна Крестителя Ибн Абу ибн Алир был снят в 1821 году и назван в честь великого визинина Ибн Мухаммеда, который был открыт в честь святого Петра I. В
wandb:                                                                                
wandb: 
wandb: Run history:
wandb:               eval/loss ▁▁
wandb:            eval/runtime █▁
wandb: eval/samples_per_second ▁█
wandb:   eval/steps_per_second ▁█
wandb:             train/epoch ▁▁▁▁▁▂▃▃▃▃▃▃▄▄▄▄▄▄▄▄▄▅▅▅▅▆▆▆▆▆▆▆▆▇▇▇████
wandb:       train/global_step ▁▁▂▂▂▂▂▂▃▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▅▅▅▅▆▆▆▆▇▇████
wandb:         train/grad_norm █▇▆▃▄▄▄▄▃▃▄▂▃▃▃▃▃▃▂▂▂▂▂▃▂▂▂▃▂▂▂▂▂▁▂▁▁▁▁▂
wandb:     train/learning_rate ▁▁██████████████████████████████████████
wandb:              train/loss █▆▆▅▄▄▄▄▄▃▃▃▃▃▃▃▃▂▂▂▂▂▂▂▂▂▂▁▁▂▂▂▁▁▁▁▁▁▁▁
wandb: 
wandb: Run summary:
wandb:                eval/loss 4.2005
wandb:             eval/runtime 47.3608
wandb:  eval/samples_per_second 105.573
wandb:    eval/steps_per_second 26.393
wandb:               total_flos 1.9008226792046592e+17
wandb:              train/epoch 0.03717
wandb:        train/global_step 3005
wandb:          train/grad_norm 0.59766
wandb:      train/learning_rate 0.0005
wandb:               train/loss 3.3971
wandb:               train_loss 4.31578
wandb:            train_runtime 1801.0722
wandb: train_samples_per_second 1077.171
wandb:   train_steps_per_second 44.882
___
В начале было слово, и слово было сделано. В результате его пребывания в течение всего дня в «Крук-кэмк» (1958—1968).

Памятники 
В 1928 году в ходе Великой Отечественной войны в ходе операции «Кабло-Боттингем Форест» была расформирована.


wandb:                                                                                
wandb: 
wandb: Run history:
wandb:               eval/loss █▁▁
wandb:            eval/runtime ▁█▆
wandb: eval/samples_per_second █▁▃
wandb:   eval/steps_per_second █▁▃
wandb:             train/epoch ▁▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▃▄▄▄▅▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇██
wandb:       train/global_step ▁▁▁▂▂▂▂▃▃▃▃▃▃▄▄▄▅▅▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇▇▇▇██
wandb:         train/grad_norm ▆▆▆▅▄▂▄▃▅▂▄█▂▂▃▄▁▂▄▂▂▂▂▁▃▂▂▁▂▁▂▁▂▂▄▂▁▁▁▁
wandb:     train/learning_rate ▁███████████████████████████████████████
wandb:              train/loss ██▄▄▄▄▄▃▃▃▂▂▂▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▂▁▁▁▁▁▂▂▂▁▁
wandb: 
wandb: Run summary:
wandb:                eval/loss 4.72866
wandb:             eval/runtime 47.8358
wandb:  eval/samples_per_second 104.524
wandb:    eval/steps_per_second 26.131
wandb:               total_flos 7.90849768610857e+16
wandb:              train/epoch 0.01547
wandb:        train/global_step 5001
wandb:          train/grad_norm 1.21094
wandb:      train/learning_rate 0.0002
wandb:               train/loss 4.1026
wandb:               train_loss 4.97192
wandb:            train_runtime 1855.1964
wandb: train_samples_per_second 1045.745
wandb:   train_steps_per_second 174.291
___
В начале было слово, и слово было сделано. В результате его пребывания в течение всего дня в «Крук-кэмк» (1958—1968).

Памятники 
В 1928 году в ходе Великой Отечественной войны в ходе операции «Кабло-Боттингем Форест» была расформирована.


wandb:                                                                                
wandb: 
wandb: Run history:
wandb:               eval/loss █▁▁
wandb:            eval/runtime ▁█▆
wandb: eval/samples_per_second █▁▃
wandb:   eval/steps_per_second █▁▃
wandb:             train/epoch ▁▁▁▁▁▂▂▂▂▂▃▃▃▃▃▃▃▄▄▄▅▅▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇██
wandb:       train/global_step ▁▁▁▂▂▂▂▃▃▃▃▃▃▄▄▄▅▅▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇▇▇▇██
wandb:         train/grad_norm ▆▆▆▅▄▂▄▃▅▂▄█▂▂▃▄▁▂▄▂▂▂▂▁▃▂▂▁▂▁▂▁▂▂▄▂▁▁▁▁
wandb:     train/learning_rate ▁███████████████████████████████████████
wandb:              train/loss ██▄▄▄▄▄▃▃▃▂▂▂▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▂▁▁▁▁▁▂▂▂▁▁
wandb: 
wandb: Run summary:
wandb:                eval/loss 4.72866
wandb:             eval/runtime 47.8358
wandb:  eval/samples_per_second 104.524
wandb:    eval/steps_per_second 26.131
wandb:               total_flos 7.90849768610857e+16
wandb:              train/epoch 0.01547
wandb:        train/global_step 5001
wandb:          train/grad_norm 1.21094
wandb:      train/learning_rate 0.0002
wandb:               train/loss 4.1026
wandb:               train_loss 4.97192
wandb:            train_runtime 1855.1964
wandb: train_samples_per_second 1045.745
wandb:   train_steps_per_second 174.291