'''Тренировочная модель для матрицы ИИ'''
data_set = {
'какой адрес у колледжа': 'Адрес - г. Бишкек, ул. Анкара (Горького), 1/17',
'по какому номеру можно связаться': 'Контактный номер WhatsApp 0707 37 99 57.',
'контактный номер': 'Контактный номер WhatsApp 0707 37 99 57.',
'инстаграм колледжа': 'Инстаграм колледжа @comtehno.kg',
'сайт колледжа': 'Сайт колледжа находится по адресу www.comtehno.kg',
'фейсбук колледжа': 'Фейсбук колледжа находится по адресу facebook.com/contehno.kg',
'зачислиться в колледж': '1.Вы делаете онлайн регистрацию. 2.Подаёте заявление на поступление. 3.Предоставляете необходимые документы. 4.Оплачиваете обучение. 5.Становитесь студентом "КОМТЕХНО". Подробнее на сайте www.comtehno.kg',
'поступить в колледж': '1.Вы делаете онлайн регистрацию. 2.Подаёте заявление на поступление. 3.Предоставляете необходимые документы. 4.Оплачиваете обучение. 5.Становитесь студентом "КОМТЕХНО". Подробнее на сайте www.comtehno.kg',
'процесс поступления в колледж': '1.Вы делаете онлайн регистрацию. 2.Подаёте заявление на поступление. 3.Предоставляете необходимые документы. 4.Оплачиваете обучение. 5.Становитесь студентом "КОМТЕХНО". Подробнее на сайте www.comtehno.kg',
'начинаются занятия': 'Занятия начинаются в 8:00 у первой смены, и в 11:00 у второй',
'в какое время начинаются занятия': 'Занятия начинаются в 8:00 у первой смены, и в 11:00 у второй',
'заканчиваются занятия': 'Занятия заканчиваются в 14:00 у первой смены, и в 17:00 у второй',
'в какое время заканчиваются занятия': 'Занятия заканчиваются в 14:00 у первой смены, и в 17:00 у второй',
'срок обучения в колледже после 9 класса': 'Нормативный срок обучения после 9 класса: 2 года 10 месяцев.',
'срок обучения в колледже после 11 класса': 'Нормативный срок обучения после 11 класса: 1 года 10 месяцев.',
'что такое сессия': 'Сессия – период сдачи экзаменов за текущий семестр. Она состоит из нескольких экзаменов. Порядок и правила сдачи похожи на сдачу школьных экзаменов. Из отметки, полученной за экзамен и отметки, полученной за две модульные недели, выводится общий итог по предмету. Именно этот балл будет выставлен на ведомость.',
'модульная неделя': 'Модульная неделя – это такая же неделя с обычным расписанием занятий. Только преподаватели в течении нее проводят контрольные, подводят некоторые итоги лабораторных занятий для промежуточного контроля знаний. Таких модульных недель в учебном семестре 2. Итоги двух модульных недель могут повлиять на отметку, с которой вы выходите на экзамен или зачет.',
'семестр': 'Семестр - это период обучения, в течение которого студенты посещают занятия по определенным предметам в учебном заведении. Обычно семестр длится полгода и делится на два учебных периода - осенний и весенний.',
'': '',
'': '',
'': '',
'какие специальности в колледже': 'Программное обеспечение вычислительной техники и автоматизированных систем, техническое обслуживание средств вычислительной техники и компьютерных сетей, прикладная информатика в разных областях, дизайн в различных отраслях, экономика и бухгалтерский учет, менеджмент, банковское дело.',
} 