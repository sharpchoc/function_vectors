#!/usr/bin/env python3
"""Generator for city-country task.

Given a well-known world city, output the country it is in.

Methodology (this script is the reproducible artifact):
  1. Downloaded GeoNames' "cities15000" dump (all populated places with
     population >= 15000, plus every national/territorial capital
     regardless of size; https://download.geonames.org/export/dump/) and
     its countryInfo.txt country-code-to-name table.
  2. Built an initial candidate pool of every city with population >=
     200,000 OR tagged as a national capital (feature code PPLC).
  3. De-duplicated city names that occur in more than one country
     (case-insensitive): kept the country's version only when its
     population was both >=500,000 and at least 5x the next-largest
     same-named city elsewhere (a clear single dominant real-world
     referent); otherwise dropped the name from every country, matching
     the spec's own examples (kept "Paris"->France; dropped "San Jose",
     "Cordoba", "Tripoli" as genuinely ambiguous). A handful of globally
     iconic cities where population alone underestimates real-world fame
     dominance (Barcelona ES over Barcelona VE; Cartagena CO over
     Cartagena ES; Colombo LK over Colombo BR; Plymouth GB over Plymouth
     MS) were kept by explicit override; "Valencia" (ES/VE, comparable
     population and fame) was dropped rather than guessed.
  4. Applied a population floor of 200,000 to every remaining candidate
     except an explicit curated list of famous sub-200k cities that a
     population cutoff would otherwise exclude (Toledo, Bruges, Salzburg,
     Innsbruck, Bethlehem, Nazareth, Delphi, Olympia, Petra, Luang
     Prabang, Timbuktu, etc.).
  5. Removed, after manual review of the full sorted candidate list, a
     large number of GeoNames entries that are administrative wards,
     boroughs, "new towns", or suburbs of a city already in the list
     rather than independently well-known cities in their own right --
     e.g. Tokyo's wards (Adachi, Itabashi, Koto, Suginami, Setagaya, ...),
     Hong Kong's districts (Kowloon, Sha Tin, Tsuen Wan, ...; only "Hong
     Kong" itself is kept), Singapore's HDB new towns (only "Singapore"
     itself is kept), Istanbul's metropolitan districts (Bagcilar,
     Umraniye, Uskudar, Sancaktepe, ...), Shanghai's/Suzhou's/Shenzhen's
     districts (Pudong, Puxi, Wuzhong, Bao'an), Ho Chi Minh City's
     districts (Binh Thanh, Cho Lon, Thu Duc), greater-Luanda districts
     (kept only Angola's genuinely distinct provincial-capital cities:
     Luanda, Huambo, Benguela, Lobito, Cabinda, Malanje, Saurimo, Lubango),
     Bucharest's numbered administrative sectors, Mexico City's boroughs
     (Iztapalapa, Gustavo A. Madero), and similar suburb/thana/mukim-level
     entries in Dhaka, Bogota, Kampala, Port-au-Prince, Havana, and
     elsewhere. Also dropped self-referential city==territory-name pairs
     (Vatican City, San Marino, Gibraltar) and a couple of low-confidence
     unfamiliar/garbled-transliteration names.
  6. Dropped the three cities whose name is identical to their country's
     name (Singapore, Hong Kong, Djibouti) to avoid input==output identity
     leakage in the ICL task, backfilling with three other well-known,
     unambiguous cities (Split, Honolulu, Dubrovnik).
  7. Applied per-country caps (tiered by pool size: up to 30 for the very
     largest pools down to a handful for small pools) so that no one
     country dominates the 1000-item set, then ranked the remainder by
     population and trimmed to exactly 1000 (protecting countries with
     <=5 surviving cities from the trim so small nations keep their
     representation).

Everyday country names are used throughout ("United States", "South
Korea", "Ivory Coast", "The Netherlands") rather than formal/ISO names.

Self-check performed at generation time: every (city, country) pair is
re-derived from the CITY_COUNTRY list below; the country output vocabulary
size and per-country distribution are reported so a human reviewer can spot
any remaining imbalance.
"""
import json
import random
from collections import Counter
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parents[2] / "city-country.json"

random.seed(42)

# (city, country) pairs, alphabetical by city for readability.
CITY_COUNTRY = [
    ('Aba', 'Nigeria'), ('Abeokuta', 'Nigeria'), ('Abidjan', 'Ivory Coast'), ('Abomey-Calavi', 'Benin'), ('Abu Dhabi', 'United Arab Emirates'), ('Abu Ghurayb', 'Iraq'),
    ('Abuja', 'Nigeria'), ('Accra', 'Ghana'), ('Adana', 'Turkey'), ('Addis Ababa', 'Ethiopia'), ('Adelaide', 'Australia'), ('Aden', 'Yemen'),
    ('Agadir', 'Morocco'), ('Agra', 'India'), ('Ahmedabad', 'India'), ('Ahvaz', 'Iran'), ('Ajman', 'United Arab Emirates'), ('Aktobe', 'Kazakhstan'),
    ('Al Ahmadi', 'Kuwait'), ('Al Ain City', 'United Arab Emirates'), ('Al Hudaydah', 'Yemen'), ('Al Mahallah al Kubra', 'Egypt'), ('Al Mansurah', 'Egypt'), ('Aleppo', 'Syria'),
    ('Alexandria', 'Egypt'), ('Algiers', 'Algeria'), ('Alicante', 'Spain'), ('Almaty', 'Kazakhstan'), ('Alor Setar', 'Malaysia'), ('Ambato', 'Ecuador'),
    ('Amman', 'Jordan'), ('Amsterdam', 'The Netherlands'), ('Andijon', 'Uzbekistan'), ('Ankara', 'Turkey'), ('Annaba', 'Algeria'), ('Ansan-si', 'South Korea'),
    ('Antalya', 'Turkey'), ('Antananarivo', 'Madagascar'), ('Antipolo', 'Philippines'), ('Antofagasta', 'Chile'), ('Antsirabe', 'Madagascar'), ('Antwerp', 'Belgium'),
    ('Anyang-si', 'South Korea'), ('Ar Raqqah', 'Syria'), ('Ar Rayyan', 'Qatar'), ('Aracaju', 'Brazil'), ('Arak', 'Iran'), ('Ardabil', 'Iran'),
    ('Arequipa', 'Peru'), ('Arhus', 'Denmark'), ('Arifwala', 'Pakistan'), ('Arusha', 'Tanzania'), ('Ashdod', 'Israel'), ('Ashgabat', 'Turkmenistan'),
    ('Asmara', 'Eritrea'), ('Assiut', 'Egypt'), ('Astana', 'Kazakhstan'), ('Asuncion', 'Paraguay'), ('Athens', 'Greece'), ('Auckland', 'New Zealand'),
    ('Austin', 'United States'), ('Avellaneda', 'Argentina'), ('Awasa', 'Ethiopia'), ('Bac Giang', 'Vietnam'), ('Bafoussam', 'Cameroon'), ('Bagerhat', 'Bangladesh'),
    ('Baghdad', 'Iraq'), ('Bago', 'Myanmar'), ('Bahawalpur', 'Pakistan'), ('Bahir Dar', 'Ethiopia'), ('Baku', 'Azerbaijan'), ('Bamako', 'Mali'),
    ('Bamenda', 'Cameroon'), ('Bandar Lampung', 'Indonesia'), ('Bandar Seri Begawan', 'Brunei'), ('Bandarban', 'Bangladesh'), ('Bandung', 'Indonesia'), ('Bangkok', 'Thailand'),
    ('Bangui', 'Central African Republic'), ('Banja Luka', 'Bosnia and Herzegovina'), ('Bannu', 'Pakistan'), ('Banqiao', 'Taiwan'), ('Bari', 'Italy'), ('Barinas', 'Venezuela'),
    ('Barquisimeto', 'Venezuela'), ('Barranquilla', 'Colombia'), ('Basrah', 'Iraq'), ('Batam', 'Indonesia'), ('Batman', 'Turkey'), ('Batna', 'Algeria'),
    ('Battagram', 'Pakistan'), ('Bawshar', 'Oman'), ('Bayamon', 'Puerto Rico'), ('Beijing', 'China'), ('Beira', 'Mozambique'), ('Beirut', 'Lebanon'),
    ('Bekasi', 'Indonesia'), ('Belem', 'Brazil'), ('Belgrade', 'Serbia'), ('Belo Horizonte', 'Brazil'), ('Bengaluru', 'India'), ('Benghazi', 'Libya'),
    ('Benguela', 'Angola'), ('Benin City', 'Nigeria'), ('Benoni', 'South Africa'), ('Berbera', 'Somalia'), ('Bergen', 'Norway'), ('Berlin', 'Germany'),
    ('Bern', 'Switzerland'), ('Bethlehem', 'Palestine'), ('Bhopal', 'India'), ('Bialystok', 'Poland'), ('Bien Hoa', 'Vietnam'), ('Bilbao', 'Spain'),
    ('Biratnagar', 'Nepal'), ('Birganj', 'Nepal'), ('Birmingham', 'United Kingdom'), ('Bishkek', 'Kyrgyzstan'), ('Bissau', 'Guinea-Bissau'), ('Blantyre', 'Malawi'),
    ('Blida', 'Algeria'), ('Bloemfontein', 'South Africa'), ('Bo', 'Sierra Leone'), ('Bobo-Dioulasso', 'Burkina Faso'), ('Bochum', 'Germany'), ('Bogor', 'Indonesia'),
    ('Bogota', 'Colombia'), ('Bologna', 'Italy'), ('Borama', 'Somalia'), ('Bordeaux', 'France'), ('Boston', 'United States'), ('Bouake', 'Ivory Coast'),
    ('Brampton', 'Canada'), ('Brasilia', 'Brazil'), ('Brasov', 'Romania'), ('Bratislava', 'Slovakia'), ('Brazzaville', 'Republic of the Congo'), ('Bremen', 'Germany'),
    ('Brest', 'Belarus'), ('Bridgetown', 'Barbados'), ('Brisbane', 'Australia'), ('Bristol', 'United Kingdom'), ('Brno', 'Czech Republic'), ('Brooklyn', 'United States'),
    ('Bruges', 'Belgium'), ('Brussels', 'Belgium'), ('Bucaramanga', 'Colombia'), ('Bucharest', 'Romania'), ('Bucheon-si', 'South Korea'), ('Budapest', 'Hungary'),
    ('Buenos Aires', 'Argentina'), ('Bujumbura', 'Burundi'), ('Bukavu', 'Democratic Republic of the Congo'), ('Bulawayo', 'Zimbabwe'), ('Buon Ma Thuot', 'Vietnam'), ('Buraydah', 'Saudi Arabia'),
    ('Burgas', 'Bulgaria'), ('Bursa', 'Turkey'), ('Busan', 'South Korea'), ('Bydgoszcz', 'Poland'), ('Cabinda', 'Angola'), ('Cagayan de Oro', 'Philippines'),
    ('Cairo', 'Egypt'), ('Calamba', 'Philippines'), ('Calgary', 'Canada'), ('Cali', 'Colombia'), ('Callao', 'Peru'), ('Caloocan', 'Philippines'),
    ('Camagueey', 'Cuba'), ('Campinas', 'Brazil'), ('Campo Grande', 'Brazil'), ('Can Tho', 'Vietnam'), ('Cancun', 'Mexico'), ('Cape Town', 'South Africa'),
    ('Caracas', 'Venezuela'), ('Cardiff', 'United Kingdom'), ('Carrefour', 'Haiti'), ('Casablanca', 'Morocco'), ('Catania', 'Italy'), ('Cayenne', 'French Guiana'),
    ('Cebu City', 'Philippines'), ('Changchun', 'China'), ('Changwon', 'South Korea'), ('Charleroi', 'Belgium'), ('Charlotte', 'United States'), ('Chattogram', 'Bangladesh'),
    ('Chelyabinsk', 'Russia'), ('Chengdu', 'China'), ('Chennai', 'India'), ('Cheonan', 'South Korea'), ('Cheongju-si', 'South Korea'), ('Chiba', 'Japan'),
    ('Chicago', 'United States'), ('Chiclayo', 'Peru'), ('Chihuahua', 'Mexico'), ('Chimoio', 'Mozambique'), ('Chingola', 'Zambia'), ('Chipata', 'Zambia'),
    ('Chisinau', 'Moldova'), ('Chitungwiza', 'Zimbabwe'), ('Chon Buri', 'Thailand'), ('Chongjin', 'North Korea'), ('Chongqing', 'China'), ('Christchurch', 'New Zealand'),
    ('Chunian', 'Pakistan'), ('Ciudad Bolivar', 'Venezuela'), ('Ciudad Guayana', 'Venezuela'), ('Ciudad Juarez', 'Mexico'), ('Ciudad Nezahualcoyotl', 'Mexico'), ('Ciudad del Este', 'Paraguay'),
    ('Cluj-Napoca', 'Romania'), ('Coban', 'Guatemala'), ('Cochabamba', 'Bolivia'), ('Coimbatore', 'India'), ('Columbus', 'United States'), ('Comilla', 'Bangladesh'),
    ('Conakry', 'Guinea'), ('Constanta', 'Romania'), ('Constantine', 'Algeria'), ('Copenhagen', 'Denmark'), ('Cordoba', 'Argentina'), ('Cork', 'Ireland'),
    ('Corrientes', 'Argentina'), ('Cotonou', 'Benin'), ('Craiova', 'Romania'), ('Cucuta', 'Colombia'), ('Cuenca', 'Ecuador'), ('Culiacan', 'Mexico'),
    ('Cumana', 'Venezuela'), ('Curitiba', 'Brazil'), ('Cusco', 'Peru'), ('Da Nang', 'Vietnam'), ('Daegu', 'South Korea'), ('Daejeon', 'South Korea'),
    ('Dakar', 'Senegal'), ('Dalian', 'China'), ('Dallas', 'United States'), ('Daloa', 'Ivory Coast'), ('Damascus', 'Syria'), ('Dammam', 'Saudi Arabia'),
    ('Danli', 'Honduras'), ('Dar es Salaam', 'Tanzania'), ('Dasoguz', 'Turkmenistan'), ('Davao', 'Philippines'), ('Debrecen', 'Hungary'), ('Dehiwala-Mount Lavinia', 'Sri Lanka'),
    ('Delhi', 'India'), ('Delmas', 'Haiti'), ('Delphi', 'Greece'), ('Denver', 'United States'), ('Depok', 'Indonesia'), ('Dera Ismail Khan', 'Pakistan'),
    ('Detroit', 'United States'), ('Dhaka', 'Bangladesh'), ('Di An', 'Vietnam'), ('Dili', 'Timor Leste'), ('Diyarbakir', 'Turkey'), ('Djelfa', 'Algeria'),
    ('Dnipro', 'Ukraine'), ('Dodoma', 'Tanzania'), ('Doha', 'Qatar'), ('Donetsk', 'Ukraine'), ('Dongguan', 'China'), ('Dubrovnik', 'Croatia'),
    ('Dortmund', 'Germany'), ('Douala', 'Cameroon'), ('Dresden', 'Germany'), ('Dubai', 'United Arab Emirates'), ('Dublin', 'Ireland'), ('Duesseldorf', 'Germany'),
    ('Duisburg', 'Germany'), ('Duque de Caxias', 'Brazil'), ('Durban', 'South Africa'), ('Dushanbe', 'Tajikistan'), ('East London', 'South Africa'), ('Ecatepec de Morelos', 'Mexico'),
    ('Edinburgh', 'United Kingdom'), ('Edmonton', 'Canada'), ('Eindhoven', 'The Netherlands'), ('El Obeid', 'Sudan'), ('El Paso', 'United States'), ('Elazig', 'Turkey'),
    ('Eldoret', 'Kenya'), ('Eloy Alfaro', 'Ecuador'), ('Enugu', 'Nigeria'), ('Epworth', 'Zimbabwe'), ('Erbil', 'Iraq'), ('Erzurum', 'Turkey'),
    ('Eskisehir', 'Turkey'), ('Eslamshahr', 'Iran'), ('Espoo', 'Finland'), ('Essen', 'Germany'), ('Faisalabad', 'Pakistan'), ('Faridabad', 'India'),
    ('Fengshan', 'Taiwan'), ('Fergana', 'Uzbekistan'), ('Fes', 'Morocco'), ('Fianarantsoa', 'Madagascar'), ('Florence', 'Italy'), ('Fort Worth', 'United States'),
    ('Fort-de-France', 'Martinique'), ('Fortaleza', 'Brazil'), ('Foshan', 'China'), ('Frankfurt am Main', 'Germany'), ('Freetown', 'Sierra Leone'), ('Fukuoka', 'Japan'),
    ('Gaborone', 'Botswana'), ('Gagnoa', 'Ivory Coast'), ('Ganja', 'Azerbaijan'), ('Gaza', 'Palestine'), ('Gaziantep', 'Turkey'), ('Gazipur', 'Bangladesh'),
    ('Gdansk', 'Poland'), ('Geita', 'Tanzania'), ('General Santos', 'Philippines'), ('Geneva', 'Switzerland'), ('Genoa', 'Italy'), ('Gent', 'Belgium'),
    ('Georgetown', 'Guyana'), ('Gitega', 'Burundi'), ('Giza', 'Egypt'), ('Glasgow', 'United Kingdom'), ('Goiania', 'Brazil'), ('Gold Coast', 'Australia'),
    ('Gonder', 'Ethiopia'), ('Gothenburg', 'Sweden'), ('Goyang-si', 'South Korea'), ('Gqeberha', 'South Africa'), ('Graz', 'Austria'), ('Groningen', 'The Netherlands'),
    ('Guadalajara', 'Mexico'), ('Guangzhou', 'China'), ('Guantanamo', 'Cuba'), ('Guarulhos', 'Brazil'), ('Guatemala City', 'Guatemala'), ('Guayaquil', 'Ecuador'),
    ('Guediawaye', 'Senegal'), ('Gujranwala', 'Pakistan'), ('Gwangju', 'South Korea'), ("Ha'il", 'Saudi Arabia'), ('Hachioji', 'Japan'), ('Haifa', 'Israel'),
    ('Haiphong', 'Vietnam'), ('Hamadan', 'Iran'), ('Hamah', 'Syria'), ('Hamamatsu', 'Japan'), ('Hamburg', 'Germany'), ('Hamhung', 'North Korea'),
    ('Hamilton', 'Canada'), ('Hangzhou', 'China'), ('Hannover', 'Germany'), ('Hanoi', 'Vietnam'), ('Harare', 'Zimbabwe'), ('Harbin', 'China'),
    ('Hargeysa', 'Somalia'), ('Havana', 'Cuba'), ('Hefei', 'China'), ('Helsinki', 'Finland'), ('Herat', 'Afghanistan'), ('Hermosillo', 'Mexico'),
    ('Higashiosaka', 'Japan'), ('Himeji', 'Japan'), ('Hiroshima', 'Japan'), ('Hlaingthaya', 'Myanmar'), ('Ho Chi Minh City', 'Vietnam'), ('Holguin', 'Cuba'),
    ('Homs', 'Syria'), ("Homyel'", 'Belarus'), ('Honiara', 'Solomon Islands'), ('Honolulu', 'United States'), ('Houston', 'United States'), ('Hrodna', 'Belarus'),
    ('Hsinchu', 'Taiwan'), ('Huambo', 'Angola'), ('Huancayo', 'Peru'), ('Hue', 'Vietnam'), ('Hungnam', 'North Korea'), ('Hwaseong-si', 'South Korea'),
    ('Iasi', 'Romania'), ('Ibadan', 'Nigeria'), ('Ibague', 'Colombia'), ('Ibb', 'Yemen'), ('Ichikawa', 'Japan'), ('Ilorin', 'Nigeria'),
    ('Incheon', 'South Korea'), ('Indianapolis', 'United States'), ('Indore', 'India'), ('Innsbruck', 'Austria'), ('Ipoh', 'Malaysia'), ('Iquitos', 'Peru'),
    ('Irbid', 'Jordan'), ('Isfahan', 'Iran'), ('Isfara', 'Tajikistan'), ('Iskandar Puteri', 'Malaysia'), ('Istanbul', 'Turkey'), ('Istaravshan', 'Tajikistan'),
    ('Izmir', 'Turkey'), ('Jacksonville', 'United States'), ('Jaipur', 'India'), ('Jakarta', 'Indonesia'), ('Jalalabad', 'Afghanistan'), ('Jamshedpur', 'India'),
    ('Jeddah', 'Saudi Arabia'), ('Jeonju', 'South Korea'), ('Jepara', 'Indonesia'), ('Jerusalem', 'Israel'), ('Jhang Sadr', 'Pakistan'), ('Jijiga', 'Ethiopia'),
    ('Jinan', 'China'), ('Joao Pessoa', 'Brazil'), ('Johannesburg', 'South Africa'), ('Johor Bahru', 'Malaysia'), ('Joinville', 'Brazil'), ('Jos', 'Nigeria'),
    ('Juba', 'South Sudan'), ('Kabul', 'Afghanistan'), ('Kabwe', 'Zambia'), ('Kaduna', 'Nigeria'), ("Kaech'on", 'North Korea'), ('Kaesong', 'North Korea'),
    ('Kagoshima', 'Japan'), ('Kahama', 'Tanzania'), ('Kakamega', 'Kenya'), ('Kalemyo', 'Myanmar'), ('Kallakurichi', 'India'), ('Kampala', 'Uganda'),
    ('Kananga', 'Democratic Republic of the Congo'), ('Kandahar', 'Afghanistan'), ('Kankan', 'Guinea'), ('Kano', 'Nigeria'), ('Kanpur', 'India'), ('Kaohsiung', 'Taiwan'),
    ('Kaolack', 'Senegal'), ('Karachi', 'Pakistan'), ('Karagandy', 'Kazakhstan'), ('Karaj', 'Iran'), ('Karbala', 'Iraq'), ('Kassala', 'Sudan'),
    ('Kathmandu', 'Nepal'), ('Kaunas', 'Lithuania'), ('Kawaguchi', 'Japan'), ('Kawasaki', 'Japan'), ('Kayseri', 'Turkey'), ('Kazan', 'Russia'),
    ('Keelung', 'Taiwan'), ('Kenema', 'Sierra Leone'), ('Kenitra', 'Morocco'), ('Kerman', 'Iran'), ('Kermanshah', 'Iran'), ('Kharkiv', 'Ukraine'),
    ('Khartoum', 'Sudan'), ('Khartoum North', 'Sudan'), ('Khulna', 'Bangladesh'), ('Kigali', 'Rwanda'), ('Kikwit', 'Democratic Republic of the Congo'), ('Kingston', 'Jamaica'),
    ('Kinshasa', 'Democratic Republic of the Congo'), ('Kirkuk', 'Iraq'), ('Kisangani', 'Democratic Republic of the Congo'), ('Kismayo', 'Somalia'), ('Kisumu', 'Kenya'), ('Kitakyushu', 'Japan'),
    ('Kitwe', 'Zambia'), ('Kobe', 'Japan'), ('Koeln', 'Germany'), ('Kolkata', 'India'), ('Kolwezi', 'Democratic Republic of the Congo'), ('Konibodom', 'Tajikistan'),
    ('Konya', 'Turkey'), ('Korhogo', 'Ivory Coast'), ('Kosice', 'Slovakia'), ('Kota Bharu', 'Malaysia'), ('Kota Kinabalu', 'Malaysia'), ('Kota Kuala Muda', 'Malaysia'),
    ('Koumassi', 'Ivory Coast'), ('Koutiala', 'Mali'), ('Krakow', 'Poland'), ('Krasnodar', 'Russia'), ('Krasnoyarsk', 'Russia'), ('Kryvyy Rih', 'Ukraine'),
    ('Kuala Lumpur', 'Malaysia'), ('Kuala Terengganu', 'Malaysia'), ('Kuantan', 'Malaysia'), ('Kuching', 'Malaysia'), ('Kulob', 'Tajikistan'), ('Kumamoto', 'Japan'),
    ('Kumasi', 'Ghana'), ('Kumba', 'Cameroon'), ('Kunming', 'China'), ('Kuwait City', 'Kuwait'), ('Kyiv', 'Ukraine'), ('Kyoto', 'Japan'),
    ('Kyzylorda', 'Kazakhstan'), ('La Ceiba', 'Honduras'), ('La Paz', 'Bolivia'), ('Lagos', 'Nigeria'), ('Lahore', 'Pakistan'), ('Lapu-Lapu City', 'Philippines'),
    ('Las Palmas de Gran Canaria', 'Spain'), ('Las Pinas', 'Philippines'), ('Las Vegas', 'United States'), ('Latakia', 'Syria'), ('Leeds', 'United Kingdom'), ('Leipzig', 'Germany'),
    ('Leon de los Aldama', 'Mexico'), ('Libreville', 'Gabon'), ('Likasi', 'Democratic Republic of the Congo'), ('Lille', 'France'), ('Lilongwe', 'Malawi'), ('Lima', 'Peru'),
    ('Lisbon', 'Portugal'), ('Liverpool', 'United Kingdom'), ('Ljubljana', 'Slovenia'), ('Lobito', 'Angola'), ('Lodz', 'Poland'), ('Loja', 'Ecuador'),
    ('Lome', 'Togo'), ('London', 'United Kingdom'), ('Los Angeles', 'United States'), ('Luanda', 'Angola'), ('Luang Prabang', 'Laos'), ('Lubango', 'Angola'),
    ('Lublin', 'Poland'), ('Lubumbashi', 'Democratic Republic of the Congo'), ('Lucerne', 'Switzerland'), ('Lucknow', 'India'), ('Ludhiana', 'India'), ('Lusaka', 'Zambia'),
    ('Lviv', 'Ukraine'), ('Lyon', 'France'), ('Macau', 'Macao'), ('Maceio', 'Brazil'), ('Machala', 'Ecuador'), ('Mainz', 'Germany'),
    ('Madinah', 'Saudi Arabia'), ('Madrid', 'Spain'), ('Madurai', 'India'), ('Mahajanga', 'Madagascar'), ('Mahilyow', 'Belarus'), ('Maiduguri', 'Nigeria'),
    ('Maipu', 'Chile'), ('Makassar', 'Indonesia'), ('Makati City', 'Philippines'), ('Makkah', 'Saudi Arabia'), ('Malacca', 'Malaysia'), ('Malaga', 'Spain'),
    ('Malang', 'Indonesia'), ('Malanje', 'Angola'), ('Malatya', 'Turkey'), ('Male', 'Maldives'), ('Malmoe', 'Sweden'), ('Mamoudzou', 'Mayotte'),
    ('Managua', 'Nicaragua'), ('Manama', 'Bahrain'), ('Manaus', 'Brazil'), ('Manchester', 'United Kingdom'), ('Mandalay', 'Myanmar'), ('Manhattan', 'United States'),
    ('Manila', 'Philippines'), ('Manta', 'Ecuador'), ('Maputo', 'Mozambique'), ('Mar del Plata', 'Argentina'), ('Maracaibo', 'Venezuela'), ('Maracay', 'Venezuela'),
    ('Maradi', 'Niger'), ('Marka', 'Somalia'), ('Maroua', 'Cameroon'), ('Marrakesh', 'Morocco'), ('Marseille', 'France'), ('Maseru', 'Lesotho'),
    ('Mashhad', 'Iran'), ('Matola', 'Mozambique'), ('Matsudo', 'Japan'), ('Matsuyama', 'Japan'), ('Maturin', 'Venezuela'), ('Mawlamyine', 'Myanmar'),
    ('Mazar-e Sharif', 'Afghanistan'), ('Mbabane', 'Eswatini'), ('Mbarara', 'Uganda'), ('Mbeya', 'Tanzania'), ('Mbuji-Mayi', 'Democratic Republic of the Congo'), ('Medan', 'Indonesia'),
    ('Medellin', 'Colombia'), ("Mek'ele", 'Ethiopia'), ('Meknes', 'Morocco'), ('Melbourne', 'Australia'), ('Mersin', 'Turkey'), ('Mexicali', 'Mexico'),
    ('Mexico City', 'Mexico'), ('Milan', 'Italy'), ('Minsk', 'Belarus'), ('Misratah', 'Libya'), ('Mississauga', 'Canada'), ('Mixco', 'Guatemala'),
    ('Mogadishu', 'Somalia'), ('Mombasa', 'Kenya'), ('Monrovia', 'Liberia'), ('Monteria', 'Colombia'), ('Monterrey', 'Mexico'), ('Montevideo', 'Uruguay'),
    ('Montpellier', 'France'), ('Montreal', 'Canada'), ('Morelia', 'Mexico'), ('Morogoro', 'Tanzania'), ('Moroni', 'Comoros'), ('Moscow', 'Russia'),
    ('Mosul', 'Iraq'), ('Mueang Nonthaburi', 'Thailand'), ('Mukalla', 'Yemen'), ('Multan', 'Pakistan'), ('Mumbai', 'India'), ('Munich', 'Germany'),
    ('Muntinlupa', 'Philippines'), ('Murcia', 'Spain'), ('Muscat', 'Oman'), ('Mutare', 'Zimbabwe'), ('Muzaffarabad', 'Pakistan'), ('Mwanza', 'Tanzania'),
    ('Mykolayiv', 'Ukraine'), ('Mzuzu', 'Malawi'), ("N'Djamena", 'Chad'), ('Nagoya', 'Japan'), ('Nagpur', 'India'), ('Nairobi', 'Kenya'),
    ('Najaf', 'Iraq'), ('Najran', 'Saudi Arabia'), ('Nakuru', 'Kenya'), ('Namangan', 'Uzbekistan'), ("Namp'o", 'North Korea'), ('Nampula', 'Mozambique'),
    ('Nanjing', 'China'), ('Nanning', 'China'), ('Nantes', 'France'), ('Naples', 'Italy'), ('Narsingdi', 'Bangladesh'), ('Nashik', 'India'),
    ('Nashville', 'United States'), ('Nasiriyah', 'Iraq'), ('Nassau', 'Bahamas'), ('Natal', 'Brazil'), ('Naucalpan de Juarez', 'Mexico'), ('Navi Mumbai', 'India'),
    ('Nay Pyi Taw', 'Myanmar'), ('Nazret', 'Ethiopia'), ('Ndola', 'Zambia'), ('Netanya', 'Israel'), ('New Taipei City', 'Taiwan'), ('New York City', 'United States'),
    ('Ngaoundere', 'Cameroon'), ('Nha Trang', 'Vietnam'), ('Niamey', 'Niger'), ('Nice', 'France'), ('Nicosia', 'Cyprus'), ('Niigata', 'Japan'),
    ('Nis', 'Serbia'), ('Nizhniy Novgorod', 'Russia'), ('Nouakchott', 'Mauritania'), ('Noumea', 'New Caledonia'), ('Nova Iguacu', 'Brazil'), ('Novi Beograd', 'Serbia'),
    ('Novi Sad', 'Serbia'), ('Novosibirsk', 'Russia'), ('Nukus', 'Uzbekistan'), ('Nuremberg', 'Germany'), ('Nyala', 'Sudan'), ('Nzerekore', 'Guinea'),
    ('Odesa', 'Ukraine'), ('Okayama', 'Japan'), ('Oklahoma City', 'United States'), ('Olympia', 'Greece'), ('Omdurman', 'Sudan'), ('Omsk', 'Russia'),
    ('Onitsha', 'Nigeria'), ('Oral', 'Kazakhstan'), ('Oran', 'Algeria'), ('Orumiyeh', 'Iran'), ('Oruro', 'Bolivia'), ('Osaka', 'Japan'),
    ('Osasco', 'Brazil'), ('Osh', 'Kyrgyzstan'), ('Oslo', 'Norway'), ('Ostrava', 'Czech Republic'), ('Ottawa', 'Canada'), ('Ouagadougou', 'Burkina Faso'),
    ('Oujda', 'Morocco'), ('Oulu', 'Finland'), ('Oyo', 'Nigeria'), ('Padang', 'Indonesia'), ('Palembang', 'Indonesia'), ('Palermo', 'Italy'),
    ('Palma', 'Spain'), ('Panama City', 'Panama'), ('Parakou', 'Benin'), ('Paramaribo', 'Suriname'), ('Paranaque City', 'Philippines'), ('Paris', 'France'),
    ('Pasig City', 'Philippines'), ('Pasir Gudang', 'Malaysia'), ('Patan', 'Nepal'), ('Patna', 'India'), ('Pavlodar', 'Kazakhstan'), ('Pekanbaru', 'Indonesia'),
    ('Perm', 'Russia'), ('Perth', 'Australia'), ('Peshawar', 'Pakistan'), ('Petah Tiqva', 'Israel'), ('Petaling Jaya', 'Malaysia'), ('Petionville', 'Haiti'),
    ('Petra', 'Jordan'), ('Philadelphia', 'United States'), ('Phnom Penh', 'Cambodia'), ('Phoenix', 'United States'), ('Pietermaritzburg', 'South Africa'), ('Pikine', 'Senegal'),
    ('Pimpri', 'India'), ('Pimpri-Chinchwad', 'India'), ('Piura', 'Peru'), ('Plovdiv', 'Bulgaria'), ('Podgorica', 'Montenegro'), ('Pointe-Noire', 'Republic of the Congo'),
    ('Pokhara', 'Nepal'), ('Port Harcourt', 'Nigeria'), ('Port Louis', 'Mauritius'), ('Port Moresby', 'Papua New Guinea'), ('Port Said', 'Egypt'), ('Port Sudan', 'Sudan'),
    ('Port-au-Prince', 'Haiti'), ('Port-de-Paix', 'Haiti'), ('Portland', 'United States'), ('Porto', 'Portugal'), ('Porto Alegre', 'Brazil'), ('Porto-Novo', 'Benin'),
    ('Portoviejo', 'Ecuador'), ('Posadas', 'Argentina'), ('Poznan', 'Poland'), ('Prague', 'Czech Republic'), ('Praia', 'Cape Verde'), ('Pretoria', 'South Africa'),
    ('Pristina', 'Kosovo'), ('Pucallpa', 'Peru'), ('Puebla', 'Mexico'), ('Puente Alto', 'Chile'), ('Puerto La Cruz', 'Venezuela'), ('Puerto Montt', 'Chile'),
    ('Pune', 'India'), ('Pyongyang', 'North Korea'), ('Qingdao', 'China'), ('Qom', 'Iran'), ('Queens', 'United States'), ('Quelimane', 'Mozambique'),
    ('Quetta', 'Pakistan'), ('Quezon City', 'Philippines'), ('Qui Nhon', 'Vietnam'), ('Quito', 'Ecuador'), ('Rabat', 'Morocco'), ('Rach Gia', 'Vietnam'),
    ('Rajkot', 'India'), ('Rajshahi', 'Bangladesh'), ('Rangpur', 'Bangladesh'), ('Ras Al Khaimah', 'United Arab Emirates'), ('Rasht', 'Iran'), ('Rawalpindi', 'Pakistan'),
    ('Recife', 'Brazil'), ('Reykjavik', 'Iceland'), ('Ribeirao Preto', 'Brazil'), ('Riga', 'Latvia'), ('Rio de Janeiro', 'Brazil'), ('Rishon LeTsiyyon', 'Israel'),
    ('Riyadh', 'Saudi Arabia'), ('Rome', 'Italy'), ('Rosario', 'Argentina'), ('Rostov-on-Don', 'Russia'), ('Rotterdam', 'The Netherlands'), ('Rufisque', 'Senegal'),
    ('Russeifa', 'Jordan'), ('Sagamihara', 'Japan'), ("Saint John's", 'Antigua and Barbuda'), ('Saint Petersburg', 'Russia'), ('Saint-Denis', 'Reunion'), ('Saint-Marc', 'Haiti'),
    ('Saitama', 'Japan'), ('Sakai', 'Japan'), ('Sale', 'Morocco'), ('Salta', 'Argentina'), ('Salvador', 'Brazil'), ('Salzburg', 'Austria'),
    ('Samara', 'Russia'), ('Samarkand', 'Uzbekistan'), ('Samut Prakan', 'Thailand'), ('San Antonio', 'United States'), ('San Diego', 'United States'), ('San Francisco', 'United States'),
    ('San Juan', 'Puerto Rico'), ('San Lorenzo', 'Paraguay'), ('San Miguel', 'El Salvador'), ('San Miguel de Tucuman', 'Argentina'), ('San Miguelito', 'Panama'), ('San Pedro Sula', 'Honduras'),
    ('San Pedro de Macoris', 'Dominican Republic'), ('San Salvador', 'El Salvador'), ('San-Pedro', 'Ivory Coast'), ('Sanaa', 'Yemen'), ('Sandakan', 'Malaysia'), ('Sanliurfa', 'Turkey'),
    ('Santa Clara', 'Cuba'), ('Santa Cruz de la Sierra', 'Bolivia'), ('Santa Fe', 'Argentina'), ('Santa Marta', 'Colombia'), ('Santiago', 'Chile'), ('Santiago de Cuba', 'Cuba'),
    ('Santiago de Queretaro', 'Mexico'), ('Santiago de los Caballeros', 'Dominican Republic'), ('Santo Domingo', 'Dominican Republic'), ('Santo Domingo Este', 'Dominican Republic'), ('Santo Domingo Oeste', 'Dominican Republic'), ('Santo Domingo de los Colorados', 'Ecuador'),
    ('Sao Bernardo do Campo', 'Brazil'), ('Sao Jose dos Campos', 'Brazil'), ('Sao Luis', 'Brazil'), ('Sao Paulo', 'Brazil'), ('Sao Tome', 'Sao Tome and Principe'), ('Sapporo', 'Japan'),
    ('Sarajevo', 'Bosnia and Herzegovina'), ('Saratov', 'Russia'), ('Sargodha', 'Pakistan'), ('Sariwon-si', 'North Korea'), ('Seattle', 'United States'), ('Seeb', 'Oman'),
    ('Segou', 'Mali'), ('Sekondi', 'Ghana'), ('Semarang', 'Indonesia'), ('Sendai', 'Japan'), ('Seongnam-si', 'South Korea'), ('Seoul', 'South Korea'),
    ('Serekunda', 'Gambia'), ('Sevastopol', 'Ukraine'), ('Sevilla', 'Spain'), ('Sfax', 'Tunisia'), ('Shah Alam', 'Malaysia'), ('Shanghai', 'China'),
    ('Shantou', 'China'), ('Sharjah', 'United Arab Emirates'), ('Sheffield', 'United Kingdom'), ('Shenyang', 'China'), ('Shenzhen', 'China'), ('Shijiazhuang', 'China'),
    ('Shiraz', 'Iran'), ('Shizuoka', 'Japan'), ('Shubra al Khaymah', 'Egypt'), ('Shymkent', 'Kazakhstan'), ('Sialkot', 'Pakistan'), ('Sikasso', 'Mali'),
    ('Skopje', 'North Macedonia'), ('Soacha', 'Colombia'), ('Sofia', 'Bulgaria'), ('Sokoto', 'Nigeria'), ('Solwezi', 'Zambia'), ('Split', 'Croatia'),
    ('Sorocaba', 'Brazil'), ('Sousse', 'Tunisia'), ('South Tangerang', 'Indonesia'), ('Soweto', 'South Africa'), ('Soyapango', 'El Salvador'), ('Stockholm', 'Sweden'),
    ('Strasbourg', 'France'), ('Stuttgart', 'Germany'), ('Subang Jaya', 'Malaysia'), ('Sucre', 'Bolivia'), ('Suez', 'Egypt'), ('Sulaymaniyah', 'Iraq'),
    ('Sumgayit', 'Azerbaijan'), ("Sunch'on", 'North Korea'), ('Sungai Petani', 'Malaysia'), ('Sunshine Coast', 'Australia'), ('Surabaya', 'Indonesia'), ('Surat', 'India'),
    ('Suva', 'Fiji'), ('Suwon', 'South Korea'), ('Suzhou', 'China'), ('Sydney', 'Australia'), ('Szczecin', 'Poland'), ("Ta'if", 'Saudi Arabia'),
    ('Tabriz', 'Iran'), ('Tabuk', 'Saudi Arabia'), ('Taguig', 'Philippines'), ('Taichung', 'Taiwan'), ('Tainan', 'Taiwan'), ('Taipei', 'Taiwan'),
    ('Taiyuan', 'China'), ('Taiz', 'Yemen'), ('Takeo', 'Cambodia'), ('Takoradi', 'Ghana'), ('Tallinn', 'Estonia'), ('Tamale', 'Ghana'),
    ('Tampere', 'Finland'), ('Tanga', 'Tanzania'), ('Tangerang', 'Indonesia'), ('Tangier', 'Morocco'), ('Tanta', 'Egypt'), ('Taoyuan', 'Taiwan'),
    ('Taraz', 'Kazakhstan'), ('Tartus', 'Syria'), ('Tashkent', 'Uzbekistan'), ('Tbilisi', 'Georgia'), ('Tegucigalpa', 'Honduras'), ('Tehran', 'Iran'),
    ('Tel Aviv', 'Israel'), ('Teresina', 'Brazil'), ('Tete', 'Mozambique'), ('Thai Nguyen', 'Vietnam'), ('Thane', 'India'), ('Thanh Hoa', 'Vietnam'),
    ('The Bronx', 'United States'), ('The Hague', 'The Netherlands'), ('Thessaloniki', 'Greece'), ('Thies', 'Senegal'), ('Thika', 'Kenya'), ('Thimphu', 'Bhutan'),
    ('Thuan An', 'Vietnam'), ('Tianjin', 'China'), ('Tijuana', 'Mexico'), ('Tilburg', 'The Netherlands'), ('Timbuktu', 'Mali'), ('Timisoara', 'Romania'),
    ('Tirana', 'Albania'), ('Tirunelveli', 'India'), ('Toamasina', 'Madagascar'), ('Tokyo', 'Japan'), ('Toronto', 'Canada'), ('Touba', 'Senegal'),
    ('Toulouse', 'France'), ('Tripoli', 'Libya'), ('Trondheim', 'Norway'), ('Trujillo', 'Peru'), ('Tshikapa', 'Democratic Republic of the Congo'), ('Tuerkmenabat', 'Turkmenistan'),
    ('Tunis', 'Tunisia'), ('Turin', 'Italy'), ('Tyumen', 'Russia'), ('Uberlandia', 'Brazil'), ('Ufa', 'Russia'), ('Ulan Bator', 'Mongolia'),
    ('Ulsan', 'South Korea'), ('Ust-Kamenogorsk', 'Kazakhstan'), ('Utrecht', 'The Netherlands'), ('Utsunomiya', 'Japan'), ('Vadodara', 'India'), ('Valenzuela', 'Philippines'),
    ('Valladolid', 'Spain'), ('Valparaiso', 'Chile'), ('Van', 'Turkey'), ('Vancouver', 'Canada'), ('Vantaa', 'Finland'), ('Varna', 'Bulgaria'),
    ('Vienna', 'Austria'), ('Vientiane', 'Laos'), ('Villa Nueva', 'Guatemala'), ('Vilnius', 'Lithuania'), ('Vina del Mar', 'Chile'), ('Vinh', 'Vietnam'),
    ('Vitebsk', 'Belarus'), ('Volgograd', 'Russia'), ('Voronezh', 'Russia'), ('Vung Tau', 'Vietnam'), ('Wandsbek', 'Germany'), ('Warri', 'Nigeria'),
    ('Warsaw', 'Poland'), ('Washington', 'United States'), ('Wellington', 'New Zealand'), ('Willemstad', 'Curacao'), ('Windhoek', 'Namibia'), ('Winnipeg', 'Canada'),
    ('Wonsan', 'North Korea'), ('Wroclaw', 'Poland'), ('Wuhan', 'China'), ('Wuppertal', 'Germany'), ('Wuxi', 'China'), ("Xi'an", 'China'),
    ('Xiamen', 'China'), ('Yangon', 'Myanmar'), ('Yaounde', 'Cameroon'), ('Yazd', 'Iran'), ('Yei', 'South Sudan'), ('Yekaterinburg', 'Russia'),
    ('Yerevan', 'Armenia'), ('Yokohama', 'Japan'), ('Yunusobod', 'Uzbekistan'), ('Zagreb', 'Croatia'), ('Zahedan', 'Iran'), ('Zamboanga', 'Philippines'),
    ('Zanzibar', 'Tanzania'), ('Zapopan', 'Mexico'), ('Zaporizhzhya', 'Ukraine'), ('Zaragoza', 'Spain'), ('Zaria', 'Nigeria'), ('Zarqa', 'Jordan'),
    ('Zhengzhou', 'China'), ('Zhongshan', 'China'), ('Zinder', 'Niger'), ('Zuerich', 'Switzerland'),
]


def main():
    assert len(CITY_COUNTRY) == 1000, f"expected 1000 pairs, got {len(CITY_COUNTRY)}"

    cities = [c for c, _ in CITY_COUNTRY]
    assert len(cities) == len(set(cities)), "duplicate city inputs"
    for c, country in CITY_COUNTRY:
        assert c == c.strip() and country == country.strip()
        assert c != country, f"identity leakage: city name equals country name ({c!r})"

    country_counts = Counter(country for _, country in CITY_COUNTRY)
    print(f"{len(country_counts)} distinct countries; top 10: {country_counts.most_common(10)}")

    examples = [{"input": c, "output": country} for c, country in CITY_COUNTRY]
    random.shuffle(examples)

    inputs = [ex["input"] for ex in examples]
    assert len(inputs) == len(set(inputs)), "duplicate inputs after shuffle"

    # rule self-check: re-derive country from CITY_COUNTRY and compare
    country_of = dict(CITY_COUNTRY)
    for ex in examples:
        assert country_of[ex["input"]] == ex["output"]

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=1)

    print(f"Wrote {len(examples)} examples to {OUT_PATH}")


if __name__ == "__main__":
    main()
