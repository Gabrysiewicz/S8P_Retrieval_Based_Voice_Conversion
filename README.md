<h1 align="center"> Retrieval-Based Voice Conversion </h1>

# Cel
Celem projektu było wytrenowanie modelu na bazie głosu oraz wykorzystywanie go do interferencji z innym istniejącym głosem, w tym przypadku głosem artysty śpiewającego jakiś utwór muzyczny. Ma to na celu utworzenie zjawiska nazywanego Cover AI, czyli podłożenie głosu jednej osoby pod głos oryginalnego artysty.

Jednym z elementów było przygotowanie zbioru danych oraz wytrenowanie modelu sztucznej inteligencji. Ten projekt skupiał się na modelach wykorzystywanych do współpracy z głosem oraz falami dźwiękowymi. Do wyszkolenia modelu potrzebne są odpowiednie pliki dźwiękowe, które muszą spełniać pewne kryteria:
- głos danej osoby,
- pliki muszą być pozbawione artefaktów,
- fragmenty zawierające ciszę muszą być krótkie lub usunięte,
- może występować tylko jeden głos,
- zalecanym formatem jest WAV,
- nazwy plików nie mogą zawierać polskich liter,
- można korzystać z jednego dużego oraz wielu małych plików,
- poza głosem nie powinny być słyszalne inne dźwięki.

Do wyszkolenia modelu wykorzystany zostanie RVC-WebUI dostępny na platformie GitHub: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI. Głównym algorytmem uczącym model będzie rmvpe_gpu, natomiast algorytmem wykorzystywanym do interferencji będzie również rmvpe. Do przygotowania zbioru danych poprosiłem swojego kolegę o nagranie kilku plików dźwiękowych oraz skorzystałem z kilku nieoficjalnie dostępnych źródeł z materiałami audio.

# Zakres
Zakres projektu obejmuje:
- nagranie lub pozyskanie plików dźwiękowych,
- oczyszczenie plików dźwiękowych z artefaktów (błędy oraz outliery),
- usunięcie ciszy we fragmentach plików lub redukcja czasu trwania ciszy,
- usunięcie fragmentów zawierających niepożądane dźwięki otoczenia (szumy, stuknięcia, ćwierkające ptaki, inne głosy),
- wyszkolenie modelu,
- dobranie odpowiednich parametrów uczących,
- pozyskanie materiałów do interferencji (czysty głos artysty),
- dobranie parametrów do interferencji,
- interferencja,
- ocena oraz fine-tuning modelu i interferencji.

# Kontekst
W pierwotnym założeniu projekt zaliczeniowy miał dotyczyć rozpoznawania obrazów z wykorzystaniem modelu YOLO. Jednakże, ze względu na moje zainteresowania oraz w uzgodnieniu z prowadzącym, tematyka mojego projektu została zmieniona na wykorzystanie sztucznej inteligencji w Coverach AI. Jest to temat, który mnie interesuje oraz w którym widzę ogromny potencjał.

# Wstęp
Istnieje wiele modeli wykorzystywanych w uczeniu na bazie plików audio, różniących się pod kątem zastosowania. Jednym z popularnych modeli jest Tacotron 2 wydany przez Google. Do procesu uczenia zbiera się mnóstwo krótkich plików audio, do których następnie przygotowuje się jeden zbiorczy plik metadata.csv, zawierający nazwę pliku oraz tekst, który jest wypowiadany w danym fragmencie. Tacotron 2 umożliwia tworzenie modeli opartych o text-to-speech, które po wpisaniu danego tekstu, czytają go głosem, na którym prowadzono uczenie. Problemem tej metody jest brak odwzorowania emocji oraz artykulacji w mowie, dlatego skorzystałem z modelu RVC (Retrieval-based Voice Conversion).

RVC (Retrieval-based Voice Conversion) to nowoczesny model konwersji głosu, który przekształca mowę jednego mówcy na mowę innego, zachowując specyficzne cechy głosowe docelowego mówcy. Dzięki zaawansowanym technikom uczenia maszynowego i głębokiego uczenia, RVC wyróżnia się skutecznością i precyzją. Model ten wykorzystuje dużą bazę danych próbek mowy, analizując cechy akustyczne mowy źródłowej i porównując je z próbkami mowy docelowego mówcy, co pozwala na naturalne i realistyczne odwzorowanie głosu.

RVC znajduje zastosowanie w wielu dziedzinach, takich jak produkcja filmowa, gry komputerowe, technologie asystentów głosowych i systemy telekomunikacyjne. Umożliwia tworzenie postaci mówiących głosem znanych aktorów oraz personalizację głosu asystentów. Jednak technologia ta wymaga odpowiednich regulacji prawnych w celu ochrony prywatności i praw autorskich. Pomimo tych wyzwań, potencjał RVC jest ogromny, a jego dalszy rozwój może przynieść jeszcze bardziej zaawansowane i zróżnicowane zastosowania.

# RVC WEBUI

Autor RVC-WebUI: `https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI`

<img src="https://github.com/Gabrysiewicz/S8P_Retrieval_Based_Voice_Conversion/blob/main/RVC-WebUI.png" />
Retrieval-based Voice Conversion WebUI to interfejs użytkownika oparty na technologii przetwarzania mowy, który umożliwia konwersję głosu na podstawie modeli wyszukiwania (retrieval-based). Główne założenie tego podejścia polega na tym, że zamiast tworzyć modele konwersji dla każdej osoby indywidualnie, korzysta się z istniejących nagrań lub danych, aby przekształcać głos jednej osoby na głos innej. Interfejs webowy umożliwia użytkownikom przesyłanie nagrań, które są następnie analizowane i konwertowane z wykorzystaniem odpowiednich algorytmów i modeli dostępnych na serwerze. Jest to przykład aplikacji, która wykorzystuje zaawansowane technologie przetwarzania sygnałów audio do tworzenia nowych funkcjonalności w obszarze interakcji głosowych i cyfrowych asystentów.

RVC-WebUI domyślnie oferuje kilka modeli:
- rmvpe,
- crepe,
- harvest,
- dio.
Ponieważ model rmvpe jest najbardziej efektywny oraz wydajny, został on wykorzystany do uczenia oraz interferencji. W dalszej części wszystkie te modele zostaną pokrótce opisane, natomiast rmvpe był głównym modelem wykorzystywanym w tym projekcie.

Repozytorium RVC-WebUI zawiera dokładne instrukcje instalacji. Wszystko powinno działać, natomiast może okazać się potrzebne doinstalowanie kilku pakietów do poprawnej obsługi karty graficznej. Jednym z takich elementów jest Nvidia CUDA Toolkit oraz biblioteki wspierające konkretną wersję toolkita. Dokładne instrukcje można znaleźć w sekcji Issues w repozytorium.

# RVC2
RVC2, czyli druga generacja Retrieval-based Voice Conversion, to zaawansowany model konwersji głosu, który buduje na sukcesie swojego poprzednika. Dzięki lepszym algorytmom, RVC2 oferuje wyższą jakość konwersji, precyzyjniej odwzorowując intonację, tembr i dynamikę głosu docelowego mówcy. Udoskonalone metody pozyskiwania i przetwarzania danych umożliwiają szybszą i dokładniejszą konwersję, co jest praktyczne w zastosowaniach na żywo. Model ten jest również bardziej elastyczny, obsługując różne style mówienia i języki, co czyni go wszechstronnym narzędziem w różnych kontekstach. Jednakże, kwestia prywatności i etyki pozostaje kluczowa dla jego odpowiedniego wykorzystania.

# RMVPE 
RMVPE (Retrieval-based Multi-view Prediction and Embedding) to zaawansowany model predykcji i osadzania danych, który wykorzystuje techniki głębokiego uczenia do analizy wieloaspektowych danych. Jego główną funkcją jest przewidywanie przyszłych wartości na podstawie złożonych wzorców w zbiorach danych, dzięki czemu może znaleźć zastosowanie w takich dziedzinach jak prognozowanie sprzedaży, analiza zachowań konsumentów czy systemy rekomendacyjne. Model RMVPE analizuje dane z różnych perspektyw (multi-view), co pozwala na bardziej precyzyjne i kompleksowe zrozumienie relacji między danymi. Wykorzystując duże bazy danych i zaawansowane algorytmy, RMVPE oferuje skuteczność i dokładność, co czyni go wartościowym narzędziem w kontekście analizy danych i sztucznej inteligencji.

# Crepe
CREPE (Convolutional Representation for Pitch Estimation) to nowoczesny model estymacji wysokości dźwięku, który wykorzystuje konwolucyjne sieci neuronowe do precyzyjnej analizy sygnałów audio. Jego głównym zadaniem jest dokładne określenie wysokości tonów w czasie rzeczywistym, co jest kluczowe w takich dziedzinach jak muzyka, mowa, oraz badania akustyczne. CREPE wyróżnia się wysoką dokładnością i efektywnością dzięki wykorzystaniu zaawansowanych technik głębokiego uczenia, które pozwalają na skuteczne przetwarzanie i interpretację złożonych sygnałów dźwiękowych. Dzięki swojej precyzji i szybkości, model ten znajduje szerokie zastosowanie w aplikacjach wymagających dokładnej analizy wysokości dźwięków.

# Harvest
Harvest to zaawansowany algorytm estymacji wysokości dźwięku, zaprojektowany do precyzyjnej analizy sygnałów audio. Jego głównym celem jest dokładne określenie wysokości tonów w nagraniach, co jest kluczowe w dziedzinach takich jak muzyka, mowa i badania akustyczne. Harvest wykorzystuje zaawansowane techniki przetwarzania sygnałów i analizę czasowo-częstotliwościową, aby zapewnić wysoką dokładność i niezawodność estymacji wysokości dźwięku. Dzięki swojej precyzji i efektywności, algorytm ten znajduje szerokie zastosowanie w różnych aplikacjach, w tym w programach do edycji muzyki, systemach rozpoznawania mowy oraz w badaniach naukowych nad akustyką i fonetyką.

# PM
PM (Parselmouth) to narzędzie do analizy i przetwarzania mowy, które wykorzystuje bibliotekę Praat, popularną wśród badaczy fonetyki i lingwistyki. PM umożliwia dokładne pomiary akustyczne, takie jak wysokość dźwięku, formanty, intensywność i spektralny przepływ energii, co jest kluczowe dla analiz mowy i dźwięku. Dzięki swojej integracji z Praat, PM pozwala na automatyzację wielu procesów analitycznych, co znacznie usprawnia pracę nad dużymi zbiorami danych. Narzędzie to jest cenione za swoją dokładność, elastyczność i możliwość dostosowania do specyficznych potrzeb badawczych, znajdując zastosowanie w takich dziedzinach jak lingwistyka, fonetyka, psychologia mowy oraz technologia mowy.

# Dio
Algorytm DIO (Distributed Input Distributed Output) jest specjalizowanym narzędziem do precyzyjnej estymacji tonów lub częstotliwości dźwięków w sygnałach audio. Opracowany w kontekście sieci neuronowych, DIO wykorzystuje techniki rozproszonego wejścia i wyjścia, co pozwala na równoczesne przetwarzanie wielu fragmentów danych, co zwiększa jego efektywność. Algorytm ten jest kluczowy w aplikacjach wymagających dokładnej analizy czasowo-częstotliwościowej, takich jak przetwarzanie mowy, analiza muzyki oraz systemy rozpoznawania mowy, zapewniając wysoką precyzję estymacji tonów i ich zmian w czasie.

# Oczyszczanie danych
Przykład danych wejściowych:
<img src="https://github.com/Gabrysiewicz/S8P_Retrieval_Based_Voice_Conversion/blob/main/audacity_przed.png" />

Wynik oczyszczonych danych:
<img src="https://github.com/Gabrysiewicz/S8P_Retrieval_Based_Voice_Conversion/blob/main/audacity_po.png" />

W efekcie uzyskano bardziej jakościowy plik, który da lepsze efekty uczenia oraz krótszy czas trwania. Jest to żmudny proces, ponieważ wymaga odsłuchania całego pliku i często kończy się potrzebą usunięcia sporej części materiału ze względu na zakłócenia. Do oczyszczania wykorzystano program Audacity oraz ffmpeg do zmiany formatu pliku na obsługiwany przez Audacity.

# Proces uczenia
Wsześniej uzyskany zbiór danych umieszczamy w dogodnym katalogu i możemy przejść do procesu uczenia. Po instalacji RVC-WebUI wywołujemy komendę python infer-web.py w katalogu z repozytorium, co uruchomi lokalny serwer z WebUI.
```
(voice-ai-38) PS X:\AI\Retrieval-based-Voice-Conversion-WebUI> python .\infer-web.py
2024-06-13 02:05:27 | INFO | configs.config | Found GPU NVIDIA GeForce RTX 3050 Laptop GPU
2024-06-13 02:05:27 | INFO | configs.config | Half-precision floating-point: True, device: cuda:0
2024-06-13 02:05:37 | INFO | __main__ | Use Language: en_US
Running on local URL:  http://0.0.0.0:7865
```

<img src="https://github.com/Gabrysiewicz/S8P_Retrieval_Based_Voice_Conversion/blob/main/train.png" />

Jeżeli wszystko jest poprawnie zainstalowane, powinna być widoczna nazwa karty graficznej. W WebUI możemy:
1. Przejść do sekcji Train,
2. Nadać nazwę modelu,
3. Określić jakość próbkowania dźwięku,
4. Ustawić pitch extraction na true (jest to wymagane w śpiewaniu, ale zbędne w zwykłej mowie),
5. Ustawić wersję RVC na 2,
6. Ustawić ilość wątków procesora według własnych potrzeb (gdy korzystamy z GPU, ta opcja nie ma znaczenia),
7. Ustawić ścieżkę do katalogu ze zbiorem danych,
8. W sekcji Step2b model uczenia ustawiamy na rmvpe_gpu lub rmvpe,
8b. Liczbę epok możemy pozostawić na 20 (zaleca się wartość między 20 a 30, dalsze zwiększanie nie przekłada się na jakość modelu),
8c. Batch size również można zwiększać, natomiast nie ma takiej potrzeby,
9. Przejść do procesu uczenia, wciskając przycisk One-trick training.

RVC najpierw wczyta wszystkie pliki, które mają odpowiedni format.
```
X:\AI\Kamyk\dataset-2/steam_07_q.wav    -> Success
X:\AI\Kamyk\dataset-2/steam_01_q.wav    -> Success
X:\AI\Kamyk\dataset-2/steam_09.wav      -> Success
X:\AI\Kamyk\dataset-2/steam_05.wav      -> Success
X:\AI\Kamyk\dataset-2/steam_06.wav      -> Success
X:\AI\Kamyk\dataset-2/steam_03.wav      -> Success
X:\AI\Kamyk\dataset-2/steam_10.wav      -> Success
X:\AI\Kamyk\dataset-2/steam_02_q.wav    -> Success
X:\AI\Kamyk\dataset-2/wiersz_q.wav      -> Success
X:\AI\Kamyk\dataset-2/steam_08.wav      -> Success
X:\AI\Kamyk\dataset-2/steam_11.wav      -> Success
```

RVC następnie podzieli nasz zbiór danych na mniejsze zbiory.
```
now-269,all-0,0_0.wav,(149, 768)
now-269,all-26,11_1.wav,(67, 768)
now-269,all-52,1_38.wav,(149, 768)
now-269,all-78,3_10.wav,(136, 768)
now-269,all-104,6_11.wav,(149, 768)
now-269,all-130,6_35.wav,(149, 768)
now-269,all-156,6_59.wav,(149, 768)
now-269,all-182,8_0.wav,(149, 768)
now-269,all-208,9_12.wav,(149, 768)
now-269,all-234,9_36.wav,(149, 768)
now-269,all-260,9_60.wav,(149, 768)
```

Finalnie przejdzie do procesu uczenia modelu, który będzie składał się z 20 epok.
Każda epoka, przy jednakowym obciążeniu sprzętu, powinna trwać mniej więcej tyle samo czasu.
Więc jeżeli pierwsza epoka zajmie 10 minut, to możemy oszacować 10 minut * 20 epok = 200 minut / 60 = 3 godziny 20 minut.
Dodatkowo, jeżeli dataset zawiera 12 minut materiału, to prawdopodobnie czas uczenia jednej epoki również zajmie 12 minut. Przynajmniej w moim przypadku i przy mojej karcie graficznej tak to przebiegało.
```
INFO:Kamyk-12M:Train Epoch: 1 [0%]
INFO:Kamyk-12M:[0, 0.0001]
INFO:Kamyk-12M:loss_disc=4.107, loss_gen=4.434, loss_fm=10.796,loss_mel=26.703, loss_kl=9.000
DEBUG:matplotlib:matplotlib data path: X:\miniconda3\envs\voice-ai-38\lib\site-packages\matplotlib\mpl-data
DEBUG:matplotlib:CONFIGDIR=C:\Users\kamil\.matplotlib
DEBUG:matplotlib:interactive is False
DEBUG:matplotlib:platform is win32
INFO:Kamyk-12M:====> Epoch: 1 [2024-06-13 02:20:09] | (0:12:58.170309)
INFO:Kamyk-12M:Train Epoch: 2 [47%]
INFO:Kamyk-12M:[200, 9.99875e-05]
INFO:Kamyk-12M:loss_disc=3.728, loss_gen=3.328, loss_fm=5.902,loss_mel=23.647, loss_kl=2.332
INFO:Kamyk-12M:====> Epoch: 2 [2024-06-13 02:33:38] | (0:13:28.865716)
INFO:Kamyk-12M:Train Epoch: 3 [94%]
INFO:Kamyk-12M:[400, 9.99750015625e-05]
INFO:Kamyk-12M:loss_disc=3.762, loss_gen=3.529, loss_fm=9.578,loss_mel=22.949, loss_kl=2.516
INFO:Kamyk-12M:====> Epoch: 3 [2024-06-13 02:46:11] | (0:12:32.941089)
INFO:Kamyk-12M:====> Epoch: 4 [2024-06-13 02:58:04] | (0:11:52.737444)
INFO:Kamyk-12M:Train Epoch: 5 [41%]
INFO:Kamyk-12M:[600, 9.995000937421877e-05]
INFO:Kamyk-12M:loss_disc=4.085, loss_gen=3.188, loss_fm=4.325,loss_mel=21.281, loss_kl=2.375
INFO:root:Saving model and optimizer state at epoch 5 to ./logs\Kamyk-12M\G_680.pth
INFO:root:Saving model and optimizer state at epoch 5 to ./logs\Kamyk-12M\D_680.pth
INFO:Kamyk-12M:====> Epoch: 5 [2024-06-13 03:10:15] | (0:12:11.262043)
INFO:Kamyk-12M:Train Epoch: 6 [88%]
INFO:Kamyk-12M:[800, 9.993751562304699e-05]
INFO:Kamyk-12M:loss_disc=3.936, loss_gen=3.558, loss_fm=9.403,loss_mel=21.352, loss_kl=1.536
INFO:Kamyk-12M:====> Epoch: 6 [2024-06-13 03:22:33] | (0:12:17.753304)
INFO:Kamyk-12M:====> Epoch: 7 [2024-06-13 03:34:33] | (0:12:00.374585)
INFO:Kamyk-12M:Train Epoch: 8 [35%]
INFO:Kamyk-12M:[1000, 9.991253280566489e-05]
INFO:Kamyk-12M:loss_disc=3.439, loss_gen=3.976, loss_fm=10.359,loss_mel=19.853, loss_kl=1.834
INFO:Kamyk-12M:====> Epoch: 8 [2024-06-13 03:46:31] | (0:11:57.633450)
INFO:Kamyk-12M:Train Epoch: 9 [82%]
INFO:Kamyk-12M:[1200, 9.990004373906418e-05]
INFO:Kamyk-12M:loss_disc=3.971, loss_gen=3.518, loss_fm=12.544,loss_mel=22.534, loss_kl=2.195
INFO:Kamyk-12M:====> Epoch: 9 [2024-06-13 03:58:36] | (0:12:04.944736)
INFO:root:Saving model and optimizer state at epoch 10 to ./logs\Kamyk-12M\G_1360.pth
INFO:root:Saving model and optimizer state at epoch 10 to ./logs\Kamyk-12M\D_1360.pth
INFO:Kamyk-12M:====> Epoch: 10 [2024-06-13 04:10:53] | (0:12:16.573370)
INFO:Kamyk-12M:Train Epoch: 11 [29%]
INFO:Kamyk-12M:[1400, 9.987507028906759e-05]
INFO:Kamyk-12M:loss_disc=4.299, loss_gen=2.925, loss_fm=9.271,loss_mel=21.341, loss_kl=1.809
INFO:Kamyk-12M:====> Epoch: 11 [2024-06-13 04:23:02] | (0:12:09.339499)
INFO:Kamyk-12M:Train Epoch: 12 [76%]
INFO:Kamyk-12M:[1600, 9.986258590528146e-05]
INFO:Kamyk-12M:loss_disc=4.257, loss_gen=3.766, loss_fm=13.422,loss_mel=22.387, loss_kl=2.152
INFO:Kamyk-12M:====> Epoch: 12 [2024-06-13 04:35:20] | (0:12:17.908730)
INFO:Kamyk-12M:====> Epoch: 13 [2024-06-13 04:47:16] | (0:11:56.373045)
INFO:Kamyk-12M:Train Epoch: 14 [24%]
INFO:Kamyk-12M:[1800, 9.983762181915804e-05]
INFO:Kamyk-12M:loss_disc=4.075, loss_gen=2.999, loss_fm=9.753,loss_mel=22.000, loss_kl=1.708
INFO:Kamyk-12M:====> Epoch: 14 [2024-06-13 04:59:00] | (0:11:43.946182)
INFO:Kamyk-12M:Train Epoch: 15 [71%]
INFO:Kamyk-12M:[2000, 9.982514211643064e-05]
INFO:Kamyk-12M:loss_disc=3.967, loss_gen=3.317, loss_fm=9.745,loss_mel=20.010, loss_kl=2.327
INFO:root:Saving model and optimizer state at epoch 15 to ./logs\Kamyk-12M\G_2040.pth
INFO:root:Saving model and optimizer state at epoch 15 to ./logs\Kamyk-12M\D_2040.pth
INFO:Kamyk-12M:====> Epoch: 15 [2024-06-13 05:10:55] | (0:11:54.278821)
INFO:Kamyk-12M:====> Epoch: 16 [2024-06-13 05:22:55] | (0:12:00.282794)
INFO:Kamyk-12M:Train Epoch: 17 [18%]
INFO:Kamyk-12M:[2200, 9.980018739066937e-05]
INFO:Kamyk-12M:loss_disc=4.610, loss_gen=3.509, loss_fm=13.866,loss_mel=21.093, loss_kl=1.551
INFO:Kamyk-12M:====> Epoch: 17 [2024-06-13 05:34:46] | (0:11:51.657800)
INFO:Kamyk-12M:Train Epoch: 18 [65%]
INFO:Kamyk-12M:[2400, 9.978771236724554e-05]
INFO:Kamyk-12M:loss_disc=4.083, loss_gen=3.007, loss_fm=6.480,loss_mel=21.605, loss_kl=2.012
INFO:Kamyk-12M:====> Epoch: 18 [2024-06-13 05:46:49] | (0:12:02.831979)
INFO:Kamyk-12M:====> Epoch: 19 [2024-06-13 05:58:47] | (0:11:57.786244)
INFO:Kamyk-12M:Train Epoch: 20 [12%]
INFO:Kamyk-12M:[2600, 9.976276699833672e-05]
INFO:Kamyk-12M:loss_disc=4.007, loss_gen=3.607, loss_fm=10.471,loss_mel=23.102, loss_kl=2.028
INFO:root:Saving model and optimizer state at epoch 20 to ./logs\Kamyk-12M\G_2720.pth
INFO:root:Saving model and optimizer state at epoch 20 to ./logs\Kamyk-12M\D_2720.pth
INFO:Kamyk-12M:====> Epoch: 20 [2024-06-13 06:10:35] | (0:11:48.248414)
INFO:Kamyk-12M:Training is done. The program is closed.
INFO:Kamyk-12M:saving final ckpt:Success.
```

# Interferencja
<img src="https://github.com/Gabrysiewicz/S8P_Retrieval_Based_Voice_Conversion/blob/main/interferencja.png" />
Wyuczony model można wykorzystać w procesie inferencji. Inferencja to zjawisko fizyczne, które polega na nakładaniu się fal, prowadzące do powstania nowego wzorca falowego. Aby stworzyć cover z użyciem AI, potrzebna będzie wersja utworu zawierająca jedynie głos, na przykład acapella. Z listy rozwijanej wybieramy nasz model i podajemy ścieżkę do wokali.

Naciśnięcie przycisku "Convert" uruchomi proces inferencji. Jeśli efekt inferencji nie jest satysfakcjonujący, można spróbować dostosować odpowiednie parametry. Mamy możliwość zmiany oktawy, na przykład obniżając ją do wartości -12, co może być pomocne, jeśli model był uczony na głosach o niskim tonie, a utwór wykonywany jest przez osobę o wysokim tonie głosu.

Resampling jest używany w przypadku, gdy model był uczony na niewielkiej liczbie próbek i konieczne jest ich zwiększenie. Zaleca się unikanie manipulowania tą opcją, ponieważ może to prowadzić do wystąpienia artefaktów. Kolejna opcja pozwala na dokładniejsze odwzorowanie względem autora oryginału (wartość 0) lub bardziej dopasowane do głosu w modelu (wartość 1). Manipulowanie tym parametrem może pomóc w osiągnięciu lepszych rezultatów.

Pozostałe parametry dotyczą ciszy, wdechów słyszalnych podczas mowy oraz akcentu. Te parametry również powinny być dostosowane w zależności od oczekiwanego rezultatu.

Otrzymany plik należy dokładnie przesłuchać. Jeśli rezultat jest satysfakcjonujący, można go pobrać i dalej przetwarzać audio.

# Podsumowanie

1. Zaczynamy od zebrania zestawu danych zawierających głos `aktora`, którego głos będziemy podkładać.
2. Następnie przygotowujemy i oczyszczamy pliki audio, co obejmuje m.in. eliminację szumów i niepożądanych artefaktów.
3. Wybieramy jeden z dostępnych algorytmów i przystępujemy do trenowania modelu, który generuje plik .pth zawierający wytrenowany model.
4. Zdobywamy plik audio piosenkarza wykonującego dany utwór (acapella lub voice only).
4b. Jeśli jest to konieczne, oczyszczamy również utwór acapella lub voice only, eliminując ewentualne zakłócenia.
5. Przeprowadzamy proces interferencji między wytrenowanym modelem a utworem acapella.
6. Wynikowy plik audio możemy dalej edytować, dodając muzykę lub dokonując innych modyfikacji w programie do obróbki audio.

<img src="https://github.com/Gabrysiewicz/S8P_Retrieval_Based_Voice_Conversion/blob/main/summary.png" />

Cały proces wymaga relatywnie: 
- Niewielkiego zbioru danych startowych.
- Czasochłonnego przygotowania danych, szczególnie oczyszczania i przygotowania plików audio.
- Dużej mocy obliczeniowej lub długiego czasu obliczeń, w zależności od wybranego algorytmu i rozmiaru danych.
- Cierpliwości i umiejętności do dobrania odpowiednich parametrów modelu i procesu interferencji.

Mając na uwadze obecne możliwości sprzętowe i ostateczny efekt, stworzenie Cover AI jest fascynującą możliwością, która może być łatwo rozwijana o dodatkowe funkcjonalności takie jak wykrywanie potencjalnych głosów nadających się do śpiewu, Real Time Translating z wykorzystanie własnego głosu, próby wykrywania deep-fake lub spoofingu. Projekt pozostawia możliwości warte rozważenia na poziomie pracy magisterskiej.





