#!/usr/bin/env python3
"""Generator for us-city-state task.

Curated (city, state) facts: well-known US cities/towns per state (capitals,
biggest cities, university towns, tourist towns), hand-picked so that a US
reader would never need the state to disambiguate the city name. Cities
whose name is shared by more than one state without one overwhelmingly
dominant referent (Springfield, Portland, Columbus, Aurora, Arlington,
Rochester, Bristol, Manchester, Salem, Franklin, Jackson, Greenville,
Georgetown, Lebanon, Athens, Cleveland (town), Concord, Auburn, Clinton,
Marion, Troy, ...) are excluded entirely.

2026-08-12 audit: a subtler tier of cross-state-ambiguous names slipped past
the original filter because this corpus itself only encodes ONE state per
name (e.g. "Laurel" was only ever entered as Delaware here, so the
cross-state-dupe check in generate() never fired, even though Laurel,
Maryland and Laurel, Mississippi are both at least as well known). See the
second block of AMBIGUOUS_NAMES below for the ~40 additional names caught
this way, each commented with its real-world competing state(s).
"""
import json
import random

CITY_STATE = [
    # Alabama
    ("Birmingham", "Alabama"), ("Montgomery", "Alabama"),
    ("Huntsville", "Alabama"), ("Mobile", "Alabama"),
    ("Tuscaloosa", "Alabama"), ("Auburn University", "Alabama"),
    ("Selma", "Alabama"), ("Gadsden", "Alabama"), ("Dothan", "Alabama"),
    ("Anniston", "Alabama"), ("Opelika", "Alabama"), ("Prattville", "Alabama"),
    ("Hoover", "Alabama"), ("Enterprise", "Alabama"), ("Talladega", "Alabama"),
    ("Fairhope", "Alabama"), ("Orange Beach", "Alabama"),
    ("Gulf Shores", "Alabama"), ("Decatur", "Alabama"), ("Bessemer", "Alabama"),
    ("Cullman", "Alabama"), ("Sylacauga", "Alabama"),
    # Alaska
    ("Anchorage", "Alaska"), ("Juneau", "Alaska"), ("Fairbanks", "Alaska"),
    ("Sitka", "Alaska"), ("Ketchikan", "Alaska"), ("Nome", "Alaska"),
    ("Wasilla", "Alaska"), ("Kodiak", "Alaska"), ("Homer", "Alaska"),
    ("Seward", "Alaska"), ("Barrow", "Alaska"), ("Palmer", "Alaska"),
    ("Skagway", "Alaska"), ("Valdez", "Alaska"),
    # Arizona
    ("Phoenix", "Arizona"), ("Tucson", "Arizona"), ("Mesa", "Arizona"),
    ("Scottsdale", "Arizona"), ("Flagstaff", "Arizona"), ("Tempe", "Arizona"),
    ("Sedona", "Arizona"), ("Yuma", "Arizona"), ("Chandler", "Arizona"),
    ("Glendale", "Arizona"), ("Gilbert", "Arizona"), ("Peoria", "Arizona"),
    ("Prescott", "Arizona"), ("Kingman", "Arizona"), ("Bisbee", "Arizona"),
    ("Lake Havasu City", "Arizona"), ("Winslow", "Arizona"),
    # Arkansas
    ("Little Rock", "Arkansas"), ("Fayetteville", "Arkansas"),
    ("Fort Smith", "Arkansas"), ("Hot Springs", "Arkansas"),
    ("Bentonville", "Arkansas"), ("Jonesboro", "Arkansas"),
    ("Conway", "Arkansas"), ("Rogers", "Arkansas"), ("Texarkana", "Arkansas"),
    ("Pine Bluff", "Arkansas"), ("Eureka Springs", "Arkansas"),
    # California
    ("Los Angeles", "California"), ("San Francisco", "California"),
    ("San Diego", "California"), ("Sacramento", "California"),
    ("San Jose", "California"), ("Fresno", "California"),
    ("Oakland", "California"), ("Long Beach", "California"),
    ("Anaheim", "California"), ("Santa Barbara", "California"),
    ("Berkeley", "California"), ("Pasadena", "California"),
    ("Palo Alto", "California"), ("Napa", "California"),
    ("Malibu", "California"), ("Bakersfield", "California"),
    ("Fremont", "California"), ("Beverly Hills", "California"),
    ("Palm Springs", "California"), ("Monterey", "California"),
    ("Santa Cruz", "California"), ("San Luis Obispo", "California"),
    ("Riverside", "California"), ("Irvine", "California"),
    ("Santa Monica", "California"), ("Chula Vista", "California"),
    ("Sunnyvale", "California"), ("Carmel-by-the-Sea", "California"),
    ("Ventura", "California"), ("Modesto", "California"),
    ("Stockton", "California"), ("Yosemite Valley", "California"),
    ("Big Sur", "California"), ("Laguna Beach", "California"),
    ("Newport Beach", "California"), ("Redwood City", "California"),
    ("Mountain View", "California"), ("Cupertino", "California"),
    # Colorado
    ("Denver", "Colorado"), ("Colorado Springs", "Colorado"),
    ("Boulder", "Colorado"), ("Aspen", "Colorado"), ("Vail", "Colorado"),
    ("Fort Collins", "Colorado"), ("Pueblo", "Colorado"),
    ("Durango", "Colorado"), ("Telluride", "Colorado"),
    ("Breckenridge", "Colorado"), ("Steamboat Springs", "Colorado"),
    ("Grand Junction", "Colorado"), ("Greeley", "Colorado"),
    ("Longmont", "Colorado"), ("Estes Park", "Colorado"), ("Golden", "Colorado"),
    # Connecticut
    ("Hartford", "Connecticut"), ("New Haven", "Connecticut"),
    ("Bridgeport", "Connecticut"), ("Stamford", "Connecticut"),
    ("Greenwich", "Connecticut"), ("Mystic", "Connecticut"),
    ("Norwalk", "Connecticut"), ("New London", "Connecticut"),
    ("Danbury", "Connecticut"), ("Waterbury", "Connecticut"),
    ("Old Saybrook", "Connecticut"),
    # Delaware
    ("Wilmington", "Delaware"), ("Dover", "Delaware"),
    ("Newark", "Delaware"), ("Rehoboth Beach", "Delaware"),
    ("Bethany Beach", "Delaware"), ("Lewes", "Delaware"),
    ("Middletown", "Delaware"),
    # Florida
    ("Miami", "Florida"), ("Orlando", "Florida"), ("Tampa", "Florida"),
    ("Jacksonville", "Florida"), ("Tallahassee", "Florida"),
    ("Fort Lauderdale", "Florida"), ("Key West", "Florida"),
    ("St. Petersburg", "Florida"), ("Sarasota", "Florida"),
    ("Naples", "Florida"), ("Pensacola", "Florida"),
    ("Gainesville", "Florida"), ("Daytona Beach", "Florida"),
    ("Boca Raton", "Florida"), ("West Palm Beach", "Florida"),
    ("Fort Myers", "Florida"), ("Clearwater", "Florida"),
    ("Panama City", "Florida"), ("St. Augustine", "Florida"),
    ("Kissimmee", "Florida"), ("Coral Gables", "Florida"),
    ("Hialeah", "Florida"), ("Palm Beach", "Florida"),
    # Georgia
    ("Atlanta", "Georgia"), ("Savannah", "Georgia"),
    ("Macon", "Georgia"), ("Marietta", "Georgia"), ("Valdosta", "Georgia"),
    ("Roswell", "Georgia"), ("Albany", "Georgia"), ("Brunswick", "Georgia"),
    ("Dalton", "Georgia"), ("Warner Robins", "Georgia"),
    ("Tybee Island", "Georgia"), ("Alpharetta", "Georgia"),
    # Hawaii
    ("Honolulu", "Hawaii"), ("Kailua", "Hawaii"),
    ("Hilo", "Hawaii"), ("Lahaina", "Hawaii"), ("Waikiki", "Hawaii"),
    ("Kaanapali", "Hawaii"), ("Kihei", "Hawaii"), ("Wailuku", "Hawaii"),
    ("Kailua-Kona", "Hawaii"), ("Haleiwa", "Hawaii"),
    # Idaho
    ("Boise", "Idaho"), ("Coeur d'Alene", "Idaho"), ("Idaho Falls", "Idaho"),
    ("Sun Valley", "Idaho"), ("Twin Falls", "Idaho"), ("Pocatello", "Idaho"),
    ("Nampa", "Idaho"), ("Moscow", "Idaho"), ("Sandpoint", "Idaho"),
    ("Ketchum", "Idaho"),
    # Illinois
    ("Chicago", "Illinois"), ("Naperville", "Illinois"),
    ("Peoria", "Illinois"), ("Rockford", "Illinois"),
    ("Evanston", "Illinois"), ("Champaign", "Illinois"),
    ("Urbana", "Illinois"), ("Joliet", "Illinois"), ("Elgin", "Illinois"),
    ("Schaumburg", "Illinois"), ("Decatur", "Illinois"),
    ("Galena", "Illinois"), ("Oak Park", "Illinois"),
    # Indiana
    ("Indianapolis", "Indiana"), ("Fort Wayne", "Indiana"),
    ("South Bend", "Indiana"), ("Gary", "Indiana"), ("Evansville", "Indiana"),
    ("West Lafayette", "Indiana"), ("Muncie", "Indiana"),
    ("Terre Haute", "Indiana"), ("Elkhart", "Indiana"),
    ("Carmel", "Indiana"), ("Bloomington, Indiana", "Indiana"),
    # Iowa
    ("Des Moines", "Iowa"), ("Cedar Rapids", "Iowa"),
    ("Iowa City", "Iowa"), ("Davenport", "Iowa"), ("Ames", "Iowa"),
    ("Waterloo", "Iowa"), ("Sioux City", "Iowa"), ("Dubuque", "Iowa"),
    ("Council Bluffs", "Iowa"),
    # Kansas
    ("Wichita", "Kansas"), ("Topeka", "Kansas"), ("Overland Park", "Kansas"),
    ("Lawrence", "Kansas"), ("Dodge City", "Kansas"), ("Salina", "Kansas"),
    ("Manhattan, Kansas", "Kansas"), ("Hutchinson", "Kansas"),
    ("Abilene, Kansas", "Kansas"),
    # Kentucky
    ("Louisville", "Kentucky"), ("Lexington", "Kentucky"),
    ("Bowling Green", "Kentucky"), ("Frankfort", "Kentucky"),
    ("Owensboro", "Kentucky"), ("Paducah", "Kentucky"),
    ("Covington", "Kentucky"), ("Berea", "Kentucky"),
    # Louisiana
    ("New Orleans", "Louisiana"), ("Baton Rouge", "Louisiana"),
    ("Shreveport", "Louisiana"), ("Lafayette", "Louisiana"),
    ("Lake Charles", "Louisiana"), ("Monroe, Louisiana", "Louisiana"),
    ("Alexandria, Louisiana", "Louisiana"), ("Houma", "Louisiana"),
    # Maine
    ("Bar Harbor", "Maine"), ("Bangor", "Maine"),
    ("Kennebunkport", "Maine"), ("Camden, Maine", "Maine"),
    ("Ogunquit", "Maine"), ("Freeport, Maine", "Maine"),
    ("Lewiston, Maine", "Maine"), ("Rockland, Maine", "Maine"),
    # Maryland
    ("Baltimore", "Maryland"), ("Annapolis", "Maryland"),
    ("Rockville", "Maryland"), ("Ocean City", "Maryland"),
    ("Frederick", "Maryland"), ("Bethesda", "Maryland"),
    ("Silver Spring", "Maryland"), ("Gaithersburg", "Maryland"),
    ("Salisbury, Maryland", "Maryland"), ("Hagerstown", "Maryland"),
    # Massachusetts
    ("Boston", "Massachusetts"), ("Cambridge", "Massachusetts"),
    ("Worcester", "Massachusetts"),
    ("Plymouth", "Massachusetts"),
    ("Nantucket", "Massachusetts"), ("Amherst", "Massachusetts"),
    ("Lowell", "Massachusetts"), ("Springfield, Massachusetts", "Massachusetts"),
    ("Provincetown", "Massachusetts"), ("Concord, Massachusetts", "Massachusetts"),
    ("Lexington, Massachusetts", "Massachusetts"), ("Quincy", "Massachusetts"),
    ("Martha's Vineyard", "Massachusetts"), ("Gloucester", "Massachusetts"),
    # Michigan
    ("Detroit", "Michigan"), ("Grand Rapids", "Michigan"),
    ("Ann Arbor", "Michigan"), ("Lansing", "Michigan"),
    ("Flint", "Michigan"), ("Traverse City", "Michigan"),
    ("Kalamazoo", "Michigan"), ("Dearborn", "Michigan"),
    ("Saginaw", "Michigan"), ("Mackinac Island", "Michigan"),
    ("East Lansing", "Michigan"), ("Marquette", "Michigan"),
    # Minnesota
    ("Minneapolis", "Minnesota"), ("Saint Paul", "Minnesota"),
    ("Duluth", "Minnesota"),
    ("Rochester, Minnesota", "Minnesota"),
    ("Bloomington, Minnesota", "Minnesota"), ("St. Cloud", "Minnesota"),
    ("Mankato", "Minnesota"), ("Brainerd", "Minnesota"),
    ("Eden Prairie", "Minnesota"), ("Winona", "Minnesota"),
    # Mississippi
    ("Jackson", "Mississippi"), ("Biloxi", "Mississippi"),
    ("Hattiesburg", "Mississippi"), ("Tupelo", "Mississippi"),
    ("Oxford, Mississippi", "Mississippi"), ("Gulfport", "Mississippi"),
    ("Natchez", "Mississippi"), ("Meridian, Mississippi", "Mississippi"),
    # Missouri
    ("St. Louis", "Missouri"), ("Kansas City", "Missouri"),
    ("Branson", "Missouri"), ("Independence, Missouri", "Missouri"),
    ("Jefferson City", "Missouri"), ("Joplin", "Missouri"),
    ("Springfield, Missouri", "Missouri"), ("Columbia, Missouri", "Missouri"),
    ("St. Charles, Missouri", "Missouri"),
    # Montana
    ("Billings", "Montana"), ("Missoula", "Montana"),
    ("Bozeman", "Montana"), ("Helena", "Montana"), ("Whitefish", "Montana"),
    ("Kalispell", "Montana"), ("Butte", "Montana"), ("Great Falls", "Montana"),
    ("Livingston, Montana", "Montana"),
    # Nebraska
    ("Omaha", "Nebraska"), ("Lincoln, Nebraska", "Nebraska"),
    ("Grand Island, Nebraska", "Nebraska"), ("Kearney, Nebraska", "Nebraska"),
    ("Norfolk, Nebraska", "Nebraska"),
    # Nevada
    ("Las Vegas", "Nevada"), ("Reno", "Nevada"), ("Carson City", "Nevada"),
    ("Henderson", "Nevada"), ("Sparks, Nevada", "Nevada"),
    ("Elko, Nevada", "Nevada"), ("Laughlin", "Nevada"),
    # New Hampshire
    ("Manchester, New Hampshire", "New Hampshire"),
    ("Concord, New Hampshire", "New Hampshire"),
    ("Nashua", "New Hampshire"), ("Portsmouth, New Hampshire", "New Hampshire"),
    ("North Conway", "New Hampshire"), ("Keene", "New Hampshire"),
    ("Hanover, New Hampshire", "New Hampshire"),
    # New Jersey
    ("Newark, New Jersey", "New Jersey"), ("Jersey City", "New Jersey"),
    ("Trenton", "New Jersey"), ("Atlantic City", "New Jersey"),
    ("Hoboken", "New Jersey"), ("Princeton", "New Jersey"),
    ("Paterson", "New Jersey"), ("Camden, New Jersey", "New Jersey"),
    ("Cape May", "New Jersey"), ("Asbury Park", "New Jersey"),
    ("Edison, New Jersey", "New Jersey"),
    # New Mexico
    ("Albuquerque", "New Mexico"), ("Santa Fe", "New Mexico"),
    ("Las Cruces", "New Mexico"), ("Taos", "New Mexico"),
    ("Roswell, New Mexico", "New Mexico"), ("Los Alamos", "New Mexico"),
    ("Farmington, New Mexico", "New Mexico"),
    # New York
    ("New York City", "New York"), ("Buffalo", "New York"),
    ("Rochester, New York", "New York"),
    ("Albany, New York", "New York"), ("Syracuse", "New York"),
    ("Yonkers", "New York"),
    ("Ithaca", "New York"), ("Niagara Falls", "New York"),
    ("Brooklyn", "New York"), ("Manhattan", "New York"),
    ("Saratoga Springs", "New York"), ("White Plains", "New York"),
    ("Poughkeepsie", "New York"), ("Utica, New York", "New York"),
    ("Montauk", "New York"), ("Woodstock, New York", "New York"),
    # North Carolina
    ("Charlotte", "North Carolina"), ("Raleigh", "North Carolina"),
    ("Durham", "North Carolina"), ("Greensboro", "North Carolina"),
    ("Asheville", "North Carolina"), ("Chapel Hill", "North Carolina"),
    ("Winston-Salem", "North Carolina"), ("Wilmington, North Carolina", "North Carolina"),
    ("Fayetteville, North Carolina", "North Carolina"), ("Boone, North Carolina", "North Carolina"),
    ("Outer Banks", "North Carolina"), ("Cary, North Carolina", "North Carolina"),
    # North Dakota
    ("Fargo", "North Dakota"), ("Bismarck", "North Dakota"),
    ("Grand Forks", "North Dakota"), ("Minot", "North Dakota"),
    ("Dickinson, North Dakota", "North Dakota"),
    # Ohio
    ("Cleveland", "Ohio"), ("Cincinnati", "Ohio"), ("Columbus, Ohio", "Ohio"),
    ("Toledo", "Ohio"), ("Akron", "Ohio"), ("Dayton", "Ohio"),
    ("Youngstown", "Ohio"), ("Canton, Ohio", "Ohio"), ("Sandusky", "Ohio"),
    ("Athens, Ohio", "Ohio"), ("Oberlin", "Ohio"), ("Marietta, Ohio", "Ohio"),
    # Oklahoma
    ("Oklahoma City", "Oklahoma"), ("Tulsa", "Oklahoma"),
    ("Norman, Oklahoma", "Oklahoma"), ("Stillwater, Oklahoma", "Oklahoma"),
    ("Broken Arrow", "Oklahoma"), ("Lawton, Oklahoma", "Oklahoma"),
    ("Enid", "Oklahoma"), ("Bartlesville", "Oklahoma"),
    # Oregon
    ("Portland, Oregon", "Oregon"), ("Eugene", "Oregon"), ("Salem, Oregon", "Oregon"),
    ("Bend, Oregon", "Oregon"), ("Ashland, Oregon", "Oregon"), ("Corvallis", "Oregon"),
    ("Astoria", "Oregon"), ("Hood River", "Oregon"), ("Medford, Oregon", "Oregon"),
    ("Newport, Oregon", "Oregon"),
    # Pennsylvania
    ("Philadelphia", "Pennsylvania"), ("Pittsburgh", "Pennsylvania"),
    ("Harrisburg", "Pennsylvania"), ("Allentown", "Pennsylvania"),
    ("Erie, Pennsylvania", "Pennsylvania"), ("Scranton", "Pennsylvania"),
    ("Gettysburg", "Pennsylvania"), ("Lancaster, Pennsylvania", "Pennsylvania"),
    ("Hershey", "Pennsylvania"), ("Bethlehem, Pennsylvania", "Pennsylvania"),
    ("Reading, Pennsylvania", "Pennsylvania"), ("State College", "Pennsylvania"),
    # Rhode Island
    ("Providence", "Rhode Island"), ("Newport, Rhode Island", "Rhode Island"),
    ("Warwick, Rhode Island", "Rhode Island"), ("Woonsocket", "Rhode Island"),
    ("Narragansett", "Rhode Island"),
    # South Carolina
    ("Charleston, South Carolina", "South Carolina"), ("Columbia, South Carolina", "South Carolina"),
    ("Greenville, South Carolina", "South Carolina"), ("Myrtle Beach", "South Carolina"),
    ("Hilton Head Island", "South Carolina"), ("Spartanburg", "South Carolina"),
    ("Beaufort, South Carolina", "South Carolina"), ("Rock Hill", "South Carolina"),
    # South Dakota
    ("Sioux Falls", "South Dakota"), ("Rapid City", "South Dakota"),
    ("Deadwood", "South Dakota"), ("Pierre", "South Dakota"),
    ("Aberdeen, South Dakota", "South Dakota"), ("Brookings, South Dakota", "South Dakota"),
    # Tennessee
    ("Nashville", "Tennessee"), ("Memphis", "Tennessee"),
    ("Knoxville", "Tennessee"), ("Chattanooga", "Tennessee"),
    ("Gatlinburg", "Tennessee"), ("Clarksville, Tennessee", "Tennessee"),
    ("Murfreesboro", "Tennessee"), ("Franklin, Tennessee", "Tennessee"),
    ("Pigeon Forge", "Tennessee"), ("Jackson, Tennessee", "Tennessee"),
    # Texas
    ("Houston", "Texas"), ("San Antonio", "Texas"), ("Dallas", "Texas"),
    ("Austin", "Texas"), ("Fort Worth", "Texas"), ("El Paso", "Texas"),
    ("Corpus Christi", "Texas"), ("Galveston", "Texas"),
    ("Lubbock", "Texas"), ("Amarillo", "Texas"), ("Waco", "Texas"),
    ("Laredo", "Texas"), ("Plano", "Texas"), ("Midland, Texas", "Texas"),
    ("Odessa, Texas", "Texas"), ("Abilene, Texas", "Texas"),
    ("Frisco, Texas", "Texas"), ("McAllen", "Texas"), ("Round Rock", "Texas"),
    ("New Braunfels", "Texas"), ("Marfa", "Texas"),
    # Utah
    ("Salt Lake City", "Utah"), ("Provo", "Utah"), ("Park City", "Utah"),
    ("Moab", "Utah"), ("Ogden", "Utah"), ("St. George, Utah", "Utah"),
    ("Logan, Utah", "Utah"), ("Orem", "Utah"),
    # Vermont
    ("Burlington, Vermont", "Vermont"), ("Montpelier", "Vermont"),
    ("Stowe", "Vermont"), ("Rutland, Vermont", "Vermont"),
    ("Brattleboro", "Vermont"), ("Woodstock, Vermont", "Vermont"),
    ("Killington", "Vermont"),
    # Virginia
    ("Richmond", "Virginia"), ("Virginia Beach", "Virginia"),
    ("Norfolk", "Virginia"),
    ("Charlottesville", "Virginia"), ("Roanoke", "Virginia"),
    ("Williamsburg", "Virginia"), ("Arlington, Virginia", "Virginia"),
    ("Alexandria, Virginia", "Virginia"), ("Chesapeake, Virginia", "Virginia"),
    ("Lynchburg, Virginia", "Virginia"), ("Fredericksburg", "Virginia"),
    # Washington
    ("Seattle", "Washington"), ("Spokane", "Washington"),
    ("Tacoma", "Washington"), ("Olympia", "Washington"),
    ("Bellevue, Washington", "Washington"), ("Vancouver, Washington", "Washington"),
    ("Bellingham", "Washington"), ("Everett, Washington", "Washington"),
    ("Yakima", "Washington"), ("Walla Walla", "Washington"),
    ("Leavenworth, Washington", "Washington"),
    # West Virginia
    ("Charleston, West Virginia", "West Virginia"), ("Huntington, West Virginia", "West Virginia"),
    ("Morgantown", "West Virginia"), ("Wheeling", "West Virginia"),
    ("Parkersburg", "West Virginia"),
    # Wisconsin
    ("Milwaukee", "Wisconsin"), ("Madison", "Wisconsin"),
    ("Green Bay", "Wisconsin"), ("Kenosha", "Wisconsin"),
    ("Racine", "Wisconsin"), ("Appleton", "Wisconsin"),
    ("Eau Claire", "Wisconsin"), ("La Crosse", "Wisconsin"),
    ("Door County", "Wisconsin"), ("Oshkosh", "Wisconsin"),
    # Wyoming
    ("Cheyenne", "Wyoming"), ("Jackson Hole", "Wyoming"),
    ("Casper", "Wyoming"), ("Laramie", "Wyoming"), ("Cody, Wyoming", "Wyoming"),
    ("Sheridan, Wyoming", "Wyoming"), ("Gillette, Wyoming", "Wyoming"),

    # ---------------- EXTRA batch: more towns per state ----------------
    ("Vestavia Hills", "Alabama"), ("Homewood, Alabama", "Alabama"),
    ("Trussville", "Alabama"), ("Foley, Alabama", "Alabama"),
    ("Daphne, Alabama", "Alabama"), ("Scottsboro", "Alabama"),
    ("Guntersville", "Alabama"), ("Muscle Shoals", "Alabama"),
    ("Eufaula, Alabama", "Alabama"),
    ("Talkeetna", "Alaska"), ("Utqiagvik", "Alaska"), ("Cordova, Alaska", "Alaska"),
    ("Haines, Alaska", "Alaska"), ("Petersburg, Alaska", "Alaska"),
    ("Unalaska", "Alaska"),
    ("Sierra Vista", "Arizona"), ("Payson, Arizona", "Arizona"),
    ("Chino Valley", "Arizona"), ("Page, Arizona", "Arizona"),
    ("Tombstone", "Arizona"), ("Show Low", "Arizona"), ("Casa Grande", "Arizona"),
    ("Mountain Home, Arkansas", "Arkansas"), ("Siloam Springs", "Arkansas"),
    ("Mena", "Arkansas"), ("Heber Springs", "Arkansas"), ("Harrison, Arkansas", "Arkansas"),
    ("Eureka, California", "California"), ("Ojai", "California"),
    ("Mendocino", "California"), ("Petaluma", "California"),
    ("Sausalito", "California"), ("Lodi, California", "California"),
    ("Visalia", "California"), ("Chico, California", "California"),
    ("Redding, California", "California"), ("Palm Desert", "California"),
    ("Solvang", "California"), ("Half Moon Bay", "California"),
    ("Avalon, California", "California"),
    ("Crested Butte", "Colorado"), ("Glenwood Springs", "Colorado"),
    ("Silverton, Colorado", "Colorado"), ("Leadville", "Colorado"),
    ("Manitou Springs", "Colorado"), ("Idaho Springs", "Colorado"),
    ("Groton", "Connecticut"), ("Litchfield", "Connecticut"),
    ("Essex, Connecticut", "Connecticut"), ("Ridgefield", "Connecticut"),
    ("New Milford", "Connecticut"),
    ("Dewey Beach", "Delaware"), ("Fenwick Island", "Delaware"),
    ("Smyrna, Delaware", "Delaware"),
    ("Cocoa Beach", "Florida"), ("Marco Island", "Florida"),
    ("Winter Park, Florida", "Florida"), ("Destin", "Florida"),
    ("Amelia Island", "Florida"), ("Vero Beach", "Florida"),
    ("Punta Gorda", "Florida"), ("Homestead, Florida", "Florida"),
    ("Weston, Florida", "Florida"), ("Islamorada", "Florida"),
    ("Kennesaw", "Georgia"), ("Sandy Springs", "Georgia"),
    ("Helen, Georgia", "Georgia"), ("St. Simons Island", "Georgia"),
    ("Jekyll Island", "Georgia"), ("LaGrange, Georgia", "Georgia"),
    ("Statesboro", "Georgia"),
    ("Princeville", "Hawaii"), ("Poipu", "Hawaii"), ("Hanalei", "Hawaii"),
    ("Wailea", "Hawaii"), ("Volcano, Hawaii", "Hawaii"),
    ("Rexburg", "Idaho"), ("Caldwell, Idaho", "Idaho"), ("Driggs", "Idaho"),
    ("McCall", "Idaho"), ("Priest Lake", "Idaho"),
    ("Waukegan", "Illinois"), ("Skokie", "Illinois"), ("DeKalb", "Illinois"),
    ("Carbondale, Illinois", "Illinois"), ("Alton, Illinois", "Illinois"),
    ("Valparaiso, Indiana", "Indiana"), ("Michigan City", "Indiana"),
    ("New Albany, Indiana", "Indiana"),
    ("Clear Lake, Iowa", "Iowa"), ("Decorah", "Iowa"), ("Pella", "Iowa"),
    ("Okoboji", "Iowa"),
    ("Emporia", "Kansas"), ("Liberal, Kansas", "Kansas"),
    ("Garden City, Kansas", "Kansas"), ("Great Bend", "Kansas"),
    ("Harrodsburg", "Kentucky"), ("Pikeville", "Kentucky"),
    ("Corbin", "Kentucky"), ("Bardstown", "Kentucky"),
    ("Natchitoches", "Louisiana"), ("Ruston", "Louisiana"),
    ("Thibodaux", "Louisiana"), ("Slidell", "Louisiana"),
    ("Opelousas", "Louisiana"),
    ("Boothbay Harbor", "Maine"), ("York, Maine", "Maine"),
    ("Belfast, Maine", "Maine"), ("Damariscotta", "Maine"),
    ("Chestertown", "Maryland"), ("St. Michaels", "Maryland"),
    ("Elkton, Maryland", "Maryland"),
    ("Falmouth, Massachusetts", "Massachusetts"), ("Sandwich, Massachusetts", "Massachusetts"),
    ("Williamstown", "Massachusetts"), ("Northampton, Massachusetts", "Massachusetts"),
    ("Stockbridge", "Massachusetts"),
    ("Petoskey", "Michigan"), ("Frankenmuth", "Michigan"),
    ("Holland, Michigan", "Michigan"), ("Charlevoix", "Michigan"),
    ("Ludington", "Michigan"),
    ("Red Wing", "Minnesota"), ("Northfield, Minnesota", "Minnesota"),
    ("Ely, Minnesota", "Minnesota"), ("Grand Marais", "Minnesota"),
    ("Vicksburg", "Mississippi"), ("Clarksdale", "Mississippi"),
    ("Starkville", "Mississippi"), ("Bay St. Louis", "Mississippi"),
    ("Hannibal, Missouri", "Missouri"), ("Sedalia", "Missouri"),
    ("Cape Girardeau", "Missouri"),
    ("Red Lodge", "Montana"), ("Big Sky", "Montana"), ("Glendive", "Montana"),
    ("Havre, Montana", "Montana"),
    ("Chadron", "Nebraska"), ("Ogallala", "Nebraska"), ("Beatrice, Nebraska", "Nebraska"),
    ("Pahrump", "Nevada"), ("Boulder City", "Nevada"),
    ("Wolfeboro", "New Hampshire"), ("Jaffrey", "New Hampshire"),
    ("Meredith, New Hampshire", "New Hampshire"),
    ("Wildwood, New Jersey", "New Jersey"), ("Long Beach Island", "New Jersey"),
    ("Montclair", "New Jersey"), ("Morristown, New Jersey", "New Jersey"),
    ("Ruidoso", "New Mexico"), ("Silver City, New Mexico", "New Mexico"),
    ("Gallup, New Mexico", "New Mexico"),
    ("Lake Placid", "New York"), ("Cooperstown", "New York"),
    ("Tarrytown", "New York"), ("Hyde Park, New York", "New York"),
    ("Watkins Glen", "New York"),
    ("Nags Head", "North Carolina"), ("Blowing Rock", "North Carolina"),
    ("Hickory, North Carolina", "North Carolina"), ("Kitty Hawk", "North Carolina"),
    ("Williston, North Dakota", "North Dakota"), ("Devils Lake", "North Dakota"),
    ("Chillicothe", "Ohio"), ("Zanesville", "Ohio"), ("Put-in-Bay", "Ohio"),
    ("Wooster", "Ohio"), ("Granville, Ohio", "Ohio"),
    ("Guthrie, Oklahoma", "Oklahoma"), ("Ardmore, Oklahoma", "Oklahoma"),
    ("Pawhuska", "Oklahoma"),
    ("Sisters, Oregon", "Oregon"), ("Cannon Beach", "Oregon"),
    ("McMinnville", "Oregon"), ("Klamath Falls", "Oregon"),
    ("Doylestown", "Pennsylvania"), ("Wilkes-Barre", "Pennsylvania"),
    ("Chambersburg", "Pennsylvania"), ("New Hope, Pennsylvania", "Pennsylvania"),
    ("Punxsutawney", "Pennsylvania"),
    ("Block Island", "Rhode Island"), ("Westerly", "Rhode Island"),
    ("Aiken", "South Carolina"), ("Sumter", "South Carolina"),
    ("Custer, South Dakota", "South Dakota"), ("Spearfish", "South Dakota"),
    ("Mitchell, South Dakota", "South Dakota"),
    ("Sevierville", "Tennessee"), ("Jonesborough", "Tennessee"),
    ("Cookeville", "Tennessee"),
    ("Kerrville", "Texas"), ("Wimberley", "Texas"), ("Port Aransas", "Texas"),
    ("South Padre Island", "Texas"), ("Nacogdoches", "Texas"),
    ("Cedar City", "Utah"), ("Springdale, Utah", "Utah"), ("Vernal", "Utah"),
    ("Middlebury", "Vermont"), ("Bennington", "Vermont"),
    ("Staunton", "Virginia"), ("Danville, Virginia", "Virginia"),
    ("Manassas", "Virginia"),
    ("Port Angeles", "Washington"), ("Ellensburg", "Washington"),
    ("Anacortes", "Washington"), ("Ocean Shores", "Washington"),
    ("Lewisburg, West Virginia", "West Virginia"), ("Berkeley Springs", "West Virginia"),
    ("Harpers Ferry", "West Virginia"), ("Beckley", "West Virginia"),
    ("Wisconsin Dells", "Wisconsin"), ("Sturgeon Bay", "Wisconsin"),
    ("Baraboo", "Wisconsin"), ("Manitowoc", "Wisconsin"),
    ("Pinedale", "Wyoming"), ("Lander, Wyoming", "Wyoming"),

    # ---------------- EXTRA batch 2: boosting under-represented states ----------------
    ("Pawtucket", "Rhode Island"), ("East Providence", "Rhode Island"),
    ("Wickford", "Rhode Island"), ("Watch Hill", "Rhode Island"),
    ("East Greenwich", "Rhode Island"), ("North Kingstown", "Rhode Island"),
    ("Nebraska City", "Nebraska"), ("Scottsbluff", "Nebraska"),
    ("Hastings, Nebraska", "Nebraska"), ("Alliance, Nebraska", "Nebraska"),
    ("Wahpeton", "North Dakota"), ("Watford City", "North Dakota"),
    ("Rugby, North Dakota", "North Dakota"), ("Mandan", "North Dakota"),
    ("Seaford, Delaware", "Delaware"), ("Laurel, Delaware", "Delaware"),
    ("Millsboro", "Delaware"),
    ("Laconia", "New Hampshire"), ("Littleton, New Hampshire", "New Hampshire"),
    ("Peterborough, New Hampshire", "New Hampshire"), ("Sunapee", "New Hampshire"),
    ("Walterboro", "South Carolina"), ("Orangeburg", "South Carolina"),
    ("Clemson", "South Carolina"), ("Bluffton, South Carolina", "South Carolina"),
    ("Shelburne", "Vermont"), ("Waitsfield", "Vermont"),
    ("St. Johnsbury", "Vermont"), ("Quechee", "Vermont"),
    ("Martinsburg", "West Virginia"), ("Elkins, West Virginia", "West Virginia"),
    ("Ripley, West Virginia", "West Virginia"), ("Shepherdstown", "West Virginia"),
    ("Mesquite, Nevada", "Nevada"), ("Winnemucca", "Nevada"),
    ("Incline Village", "Nevada"), ("Virginia City", "Nevada"),
    ("Truth or Consequences", "New Mexico"), ("Espanola", "New Mexico"),
    ("Clovis, New Mexico", "New Mexico"), ("Deming", "New Mexico"),
    ("Vermillion, South Dakota", "South Dakota"), ("Yankton", "South Dakota"),
    ("Belle Fourche", "South Dakota"), ("Sturgis, South Dakota", "South Dakota"),
    ("Thermopolis", "Wyoming"), ("Rock Springs, Wyoming", "Wyoming"),
    ("Douglas, Wyoming", "Wyoming"), ("Torrington, Wyoming", "Wyoming"),
    ("Rolla", "Missouri"), ("West Plains", "Missouri"),
    ("Kirksville", "Missouri"),
    ("Dodge City, Kansas", "Kansas"), ("Pittsburg, Kansas", "Kansas"),
    ("McPherson", "Kansas"),
    ("Middlesboro", "Kentucky"), ("Elizabethtown, Kentucky", "Kentucky"),
    ("Somerset, Kentucky", "Kentucky"),
    ("Bath, Maine", "Maine"), ("Skowhegan", "Maine"),
    ("Cleveland, Mississippi", "Mississippi"), ("Philadelphia, Mississippi", "Mississippi"),
    ("Ada, Oklahoma", "Oklahoma"), ("Poteau", "Oklahoma"),
    ("Baker City", "Oregon"), ("Pendleton, Oregon", "Oregon"),
    ("Sevierville, Tennessee", "Tennessee"), ("Kingsport", "Tennessee"),
    ("Logan, Utah", "Utah"), ("Tooele", "Utah"),
    ("Big Stone Gap", "Virginia"), ("Wytheville", "Virginia"),

    # ---------------- EXTRA batch 3: broad final top-up ----------------
    ("Blytheville", "Arkansas"), ("Arkadelphia", "Arkansas"),
    ("Batesville, Arkansas", "Arkansas"), ("Malvern, Arkansas", "Arkansas"),
    ("Russellville, Arkansas", "Arkansas"),
    ("Simsbury", "Connecticut"), ("Old Lyme", "Connecticut"),
    ("Guilford, Connecticut", "Connecticut"),
    ("Milton, Delaware", "Delaware"), ("Bridgeville, Delaware", "Delaware"),
    ("Ocean View, Delaware", "Delaware"), ("Harrington, Delaware", "Delaware"),
    ("Newnan", "Georgia"), ("Peachtree City", "Georgia"),
    ("Thomasville, Georgia", "Georgia"), ("Dahlonega", "Georgia"),
    ("Salmon, Idaho", "Idaho"), ("Rigby", "Idaho"),
    ("St. Anthony, Idaho", "Idaho"), ("Weiser", "Idaho"),
    ("Belleville, Illinois", "Illinois"), ("Effingham", "Illinois"),
    ("Kankakee", "Illinois"), ("Moline", "Illinois"),
    ("Anderson, Indiana", "Indiana"), ("Kokomo", "Indiana"),
    ("Vincennes", "Indiana"), ("Crawfordsville", "Indiana"),
    ("Mason City", "Iowa"), ("Fort Dodge", "Iowa"),
    ("Marshalltown", "Iowa"), ("Le Mars", "Iowa"),
    ("Leavenworth, Kansas", "Kansas"), ("Junction City, Kansas", "Kansas"),
    ("Winfield, Kansas", "Kansas"),
    ("Radcliff", "Kentucky"), ("Hazard, Kentucky", "Kentucky"),
    ("Cynthiana", "Kentucky"),
    ("Bossier City", "Louisiana"), ("Morgan City", "Louisiana"),
    ("Bogalusa", "Louisiana"), ("Denham Springs", "Louisiana"),
    ("Machias", "Maine"), ("Eastport, Maine", "Maine"),
    ("Wiscasset", "Maine"), ("Millinocket", "Maine"),
    ("Easton, Maryland", "Maryland"), ("Cumberland, Maryland", "Maryland"),
    ("La Plata, Maryland", "Maryland"), ("Crisfield", "Maryland"),
    ("Bemidji", "Minnesota"), ("Fergus Falls", "Minnesota"),
    ("Owatonna", "Minnesota"), ("Hibbing", "Minnesota"),
    ("Corinth, Mississippi", "Mississippi"), ("Yazoo City", "Mississippi"),
    ("Indianola, Mississippi", "Mississippi"), ("Greenwood, Mississippi", "Mississippi"),
    ("Poplar Bluff", "Missouri"), ("Warrensburg", "Missouri"),
    ("Sikeston", "Missouri"),
    ("Anaconda", "Montana"), ("Dillon, Montana", "Montana"),
    ("Wolf Point", "Montana"), ("Miles City", "Montana"),
    ("North Platte", "Nebraska"), ("Gering", "Nebraska"), ("McCook", "Nebraska"),
    ("Fallon, Nevada", "Nevada"), ("Fernley", "Nevada"), ("Yerington", "Nevada"),
    ("Bretton Woods", "New Hampshire"), ("Franconia", "New Hampshire"),
    ("Vineland", "New Jersey"), ("Toms River", "New Jersey"),
    ("Ridgewood, New Jersey", "New Jersey"),
    ("Alamogordo", "New Mexico"), ("Los Lunas", "New Mexico"),
    ("Belen, New Mexico", "New Mexico"),
    ("Beulah, North Dakota", "North Dakota"), ("Valley City", "North Dakota"),
    ("Bottineau", "North Dakota"),
    ("Lima, Ohio", "Ohio"), ("Ashtabula", "Ohio"), ("Piqua", "Ohio"),
    ("Xenia", "Ohio"),
    ("Chickasha", "Oklahoma"), ("Durant, Oklahoma", "Oklahoma"),
    ("Vinita", "Oklahoma"), ("McAlester", "Oklahoma"),
    ("Coos Bay", "Oregon"), ("Tillamook", "Oregon"),
    ("Grants Pass", "Oregon"), ("La Grande", "Oregon"),
    ("West Warwick", "Rhode Island"), ("Cranston", "Rhode Island"),
    ("Barrington, Rhode Island", "Rhode Island"),
    ("Hartsville", "South Carolina"), ("Newberry, South Carolina", "South Carolina"),
    ("Huron", "South Dakota"), ("Milbank", "South Dakota"),
    ("Tullahoma", "Tennessee"), ("Elizabethton", "Tennessee"),
    ("Crossville", "Tennessee"),
    ("Layton", "Utah"), ("Bountiful", "Utah"), ("Heber City", "Utah"),
    ("Barre", "Vermont"), ("Randolph, Vermont", "Vermont"),
    ("Suffolk, Virginia", "Virginia"), ("Blacksburg", "Virginia"),
    ("Front Royal", "Virginia"), ("Abingdon, Virginia", "Virginia"),
    ("Wenatchee", "Washington"), ("Pullman", "Washington"),
    ("Chelan", "Washington"), ("Poulsbo", "Washington"),
    ("Bluefield, West Virginia", "West Virginia"), ("Summersville", "West Virginia"),
    ("Weirton", "West Virginia"),
    ("Wausau", "Wisconsin"), ("Sheboygan", "Wisconsin"),
    ("Fond du Lac", "Wisconsin"),
    ("Green River, Wyoming", "Wyoming"), ("Powell, Wyoming", "Wyoming"),
    ("Riverton, Wyoming", "Wyoming"),

    # ---------------- EXTRA batch 4: final top-up ----------------
    ("Ord, Nebraska", "Nebraska"), ("Blair, Nebraska", "Nebraska"),
    ("Broken Bow, Nebraska", "Nebraska"),
    ("Tilton", "New Hampshire"), ("Ossipee", "New Hampshire"),
    ("Center Harbor", "New Hampshire"),
    ("Hendersonville", "North Carolina"), ("Southern Pines", "North Carolina"),
    ("Manteo", "North Carolina"),
    ("Hettinger", "North Dakota"), ("Cavalier", "North Dakota"),
    ("Walhalla, South Carolina", "South Carolina"), ("Abbeville, South Carolina", "South Carolina"),
    ("Bradford, Vermont", "Vermont"), ("Enosburg Falls", "Vermont"),
    ("Selbyville", "Delaware"), ("Frankford, Delaware", "Delaware"),
    ("Waimea", "Hawaii"), ("Hana", "Hawaii"),
    ("Colby, Kansas", "Kansas"), ("Ulysses, Kansas", "Kansas"),
    ("Kosciusko", "Mississippi"), ("Senatobia", "Mississippi"),
    ("Chepachet", "Rhode Island"), ("Foster, Rhode Island", "Rhode Island"),
    ("Redfield, South Dakota", "South Dakota"), ("Philip, South Dakota", "South Dakota"),
    ("Dyersburg", "Tennessee"), ("Lawrenceburg, Tennessee", "Tennessee"),
    ("Selmer", "Tennessee"),
    ("Manti", "Utah"), ("Nephi", "Utah"),
    ("Buckhannon", "West Virginia"), ("Grafton, West Virginia", "West Virginia"),
    ("Moorefield", "West Virginia"),
    ("Minden, Louisiana", "Louisiana"), ("Winnsboro, Louisiana", "Louisiana"),
    ("Ville Platte", "Louisiana"),
    ("Great Barrington", "Massachusetts"), ("Chatham, Massachusetts", "Massachusetts"),
    ("Rockport, Massachusetts", "Massachusetts"),
    ("Willmar", "Minnesota"), ("Worthington, Minnesota", "Minnesota"),
    ("Marshall, Missouri", "Missouri"), ("Carthage, Missouri", "Missouri"),
    ("Tonopah", "Nevada"), ("Overton, Nevada", "Nevada"),
    ("Lambertville", "New Jersey"), ("Point Pleasant, New Jersey", "New Jersey"),
    ("Spring Lake, New Jersey", "New Jersey"),
    ("Tucumcari", "New Mexico"), ("Portales", "New Mexico"),
    ("Kutztown", "Pennsylvania"), ("Jim Thorpe", "Pennsylvania"),
    ("Ephrata, Pennsylvania", "Pennsylvania"),
    ("Kemmerer", "Wyoming"), ("Sundance, Wyoming", "Wyoming"),
    ("Seymour, Indiana", "Indiana"), ("Logansport", "Indiana"),
    ("Wabash", "Indiana"),

    # ---------------- EXTRA batch 5: last top-up ----------------
    ("Lisbon, North Dakota", "North Dakota"), ("Carrington, North Dakota", "North Dakota"),
    ("Manning, South Carolina", "South Carolina"),
    ("Vergennes", "Vermont"), ("Bellows Falls", "Vermont"),
    ("Clayton, Delaware", "Delaware"),
    ("Kapaa", "Hawaii"), ("Pahoa", "Hawaii"),
    ("Spencer, Iowa", "Iowa"), ("Storm Lake", "Iowa"),
    ("Ottawa, Kansas", "Kansas"), ("Chanute", "Kansas"),
    ("Mount Sterling", "Kentucky"), ("Shelbyville, Kentucky", "Kentucky"),
    ("Presque Isle", "Maine"), ("Milbridge", "Maine"),
    ("Denton, Maryland", "Maryland"), ("Pocomoke City", "Maryland"),
    ("Alpena", "Michigan"), ("Escanaba", "Michigan"),
    ("Brookhaven, Mississippi", "Mississippi"), ("Picayune", "Mississippi"),
    ("Sidney, Montana", "Montana"), ("Deer Lodge", "Montana"),
    ("Wayne, Nebraska", "Nebraska"), ("Superior, Nebraska", "Nebraska"),
    ("Colebrook", "New Hampshire"), ("Whitefield, New Hampshire", "New Hampshire"),
    ("Elmira", "New York"), ("Oswego", "New York"), ("Canandaigua", "New York"),
    ("Morehead City", "North Carolina"), ("Sanford, North Carolina", "North Carolina"),
    ("Mansfield, Ohio", "Ohio"), ("Ironton", "Ohio"),
    ("Sallisaw", "Oklahoma"), ("Woodward, Oklahoma", "Oklahoma"),
    ("Roseburg", "Oregon"), ("John Day", "Oregon"),

    # ---------------- EXTRA batch 6: replenish after 2026-08-12 ambiguity
    # audit (see AMBIGUOUS_NAMES "subtler tier" above) -- distinctive,
    # single-state-identity towns, weighted toward the states that lost the
    # most entries (Maryland, Mississippi, Virginia, Arkansas). ----------------
    ("Havre de Grace", "Maryland"), ("Snow Hill", "Maryland"),
    ("Princess Anne", "Maryland"), ("Federalsburg", "Maryland"),
    ("Ellicott City", "Maryland"), ("Towson", "Maryland"),
    ("Catonsville", "Maryland"), ("North East, Maryland", "Maryland"),
    ("Pikesville", "Maryland"), ("Perryville, Maryland", "Maryland"),
    ("Pascagoula", "Mississippi"), ("Ocean Springs", "Mississippi"),
    ("Bay Springs", "Mississippi"), ("Ridgeland, Mississippi", "Mississippi"),
    ("Flowood", "Mississippi"), ("Grenada, Mississippi", "Mississippi"),
    ("Amory", "Mississippi"), ("Byhalia", "Mississippi"),
    ("Pontotoc", "Mississippi"), ("Booneville, Mississippi", "Mississippi"),
    ("Martinsville", "Virginia"), ("Galax", "Virginia"),
    ("Pulaski, Virginia", "Virginia"), ("South Boston, Virginia", "Virginia"),
    ("Farmville", "Virginia"), ("Christiansburg", "Virginia"),
    ("Radford", "Virginia"), ("Pearisburg", "Virginia"),
    ("West Memphis", "Arkansas"), ("Stuttgart, Arkansas", "Arkansas"),
    ("Searcy", "Arkansas"), ("Forrest City", "Arkansas"),
    ("Cabot, Arkansas", "Arkansas"), ("Sherwood, Arkansas", "Arkansas"),
    ("Trumann", "Arkansas"), ("Osceola, Arkansas", "Arkansas"),
    ("Fairmont, West Virginia", "West Virginia"),
    ("Clarksburg, West Virginia", "West Virginia"),
    ("Oak Hill, West Virginia", "West Virginia"),
    ("Hinton, West Virginia", "West Virginia"),
    ("Welch, West Virginia", "West Virginia"),
    ("Blades", "Delaware"), ("Delaware City", "Delaware"),
    ("Townsend, Delaware", "Delaware"), ("Felton, Delaware", "Delaware"),
    ("Kenner", "Louisiana"), ("Mandeville, Louisiana", "Louisiana"),
    ("Zachary", "Louisiana"), ("Eunice, Louisiana", "Louisiana"),
    ("Chalmette", "Louisiana"),
    ("Concordia, Kansas", "Kansas"), ("Iola", "Kansas"),
    ("Coffeyville", "Kansas"), ("Parsons, Kansas", "Kansas"),
    ("Larned", "Kansas"),
    ("Dillingham", "Alaska"),
    ("Buckeye", "Arizona"), ("Oro Valley", "Arizona"),
    ("Milledgeville", "Georgia"), ("Cordele", "Georgia"),
    ("Douglasville", "Georgia"),
    ("Portage, Indiana", "Indiana"), ("Warsaw, Indiana", "Indiana"),
    ("Plainfield, Indiana", "Indiana"),
    ("London, Kentucky", "Kentucky"), ("Harlan", "Kentucky"),
    ("Barbourville", "Kentucky"),
    ("Fort Kent", "Maine"), ("Norway, Maine", "Maine"),
    ("Fitchburg", "Massachusetts"), ("Leominster", "Massachusetts"),
    ("Pittsfield", "Massachusetts"), ("Attleboro", "Massachusetts"),
    ("Bolivar, Missouri", "Missouri"), ("Nixa", "Missouri"),
    ("David City", "Nebraska"), ("Gothenburg, Nebraska", "Nebraska"),
    ("Exeter, New Hampshire", "New Hampshire"),
    ("Derry, New Hampshire", "New Hampshire"),
    ("Hobbs", "New Mexico"), ("Artesia, New Mexico", "New Mexico"),
    ("Raton", "New Mexico"),
    ("Duncan, Oklahoma", "Oklahoma"), ("Elk City", "Oklahoma"),
    ("Lincoln City", "Oregon"), ("Sweet Home, Oregon", "Oregon"),
    ("Meadville", "Pennsylvania"), ("Carlisle, Pennsylvania", "Pennsylvania"),
    ("Seneca, South Carolina", "South Carolina"), ("Easley", "South Carolina"),
    ("Lead, South Dakota", "South Dakota"), ("Winner, South Dakota", "South Dakota"),
    ("Sugar Land", "Texas"), ("Cedar Park, Texas", "Texas"),
    ("Kanab", "Utah"), ("Brigham City", "Utah"),
]

# Ambiguous-name filter: names shared by cities in more than one state
# without an overwhelmingly dominant referent are dropped entirely, even
# though they appear in the raw list above (some intentionally kept because
# there IS a clearly dominant referent -- e.g. Columbus, Ohio; Portland,
# Oregon is genuinely ambiguous with Portland, Maine so BOTH are dropped).
AMBIGUOUS_NAMES = {
    "Springfield", "Portland", "Columbia", "Salem", "Athens", "Arlington",
    "Alexandria", "Charleston", "Franklin", "Georgetown", "Lebanon",
    "Bloomington", "Auburn", "Clinton", "Marion", "Troy", "Manchester",
    "Rochester", "Bristol", "Newark", "Ontario", "Vancouver",
    "Wilmington", "Columbus", "Augusta",

    # ---- subtler tier (2026-08-12 audit): the name in this corpus resolves
    # to only ONE state (so the cross-state-dupe check below never sees it),
    # but there is an at-least-comparably-known city with the exact same
    # name in another state, so a US reader would still need the state to
    # disambiguate. Each entry names the competing state(s).
    "Aberdeen",       # WA (Kurt Cobain/Nirvana hometown) vs corpus's SD
    "Amherst",        # NY (Buffalo suburb, larger than MA's) vs corpus's MA
    "Berea",          # OH (Baldwin Wallace, ex-Browns camp) vs corpus's KY
    "Bluefield",      # VA (literal twin city across the state line) vs corpus's WV
    "Bowling Green",  # OH (Bowling Green State Univ.) vs corpus's KY
    "Brookhaven",     # NY (huge Long Island township, Brookhaven Nat'l Lab) vs corpus's MS
    "Brunswick",      # ME (Bowdoin College) / OH vs corpus's GA
    "Carmel",         # CA (Carmel-by-the-Sea) vs corpus's IN
    "Clovis",         # CA (Fresno suburb, larger than NM's) vs corpus's NM
    "Conway",         # SC (Coastal Carolina Univ.) vs corpus's AR
    "Danville",       # CA (affluent SF Bay suburb) vs corpus's VA
    "Denton",         # TX (much larger/better known) vs corpus's MD
    "Easton",         # PA (Lafayette College, Crayola) vs corpus's MD
    "Fredericksburg", # TX (Hill Country tourist town) vs corpus's VA
    "Garden City",    # NY (Long Island) vs corpus's KS
    "Glendale",       # CA (comparable size/fame) vs corpus's AZ
    "Greenville",     # NC (East Carolina Univ.) / MS vs corpus's SC
    "Huntington",     # NY (Long Island) vs corpus's WV
    "Indianola",      # IA (Simpson College) vs corpus's MS
    "Jonesboro",      # GA (Gone with the Wind lore) vs corpus's AR
    "Kansas City",    # KS (literal twin city across the state line) vs corpus's MO
    "Lafayette",      # IN (Purdue-area twin city) vs corpus's LA
    "Lancaster",      # CA (larger population) vs corpus's PA
    "Laurel",         # MD / MS (both well known) vs corpus's DE
    "Lawrence",       # MA (historic mill city) vs corpus's KS
    "Lynchburg",      # TN (Jack Daniel's, globally printed on the label) vs corpus's VA
    "Medford",        # MA (Tufts Univ.) vs corpus's OR
    "Meridian",       # ID (large, fast-growing Boise suburb) vs corpus's MS
    "Midland",        # MI (Dow Chemical HQ) vs corpus's TX
    "Milton",         # MA / GA (both well known) vs corpus's DE
    "Monroe",         # MI / NC (Charlotte suburb) vs corpus's LA
    "Ocean City",     # NJ (comparable beach resort) vs corpus's MD
    "Oxford",         # OH (Miami University) vs corpus's MS
    "Petersburg",     # VA (Civil War siege, much better known) vs corpus's AK
    "Portsmouth",     # VA (Hampton Roads) vs corpus's NH
    "Salisbury",      # NC (comparable size/fame) vs corpus's MD
    "Springdale",     # AR (large NWA city) vs corpus's UT
    "Stillwater",     # MN (historic river town) vs corpus's OK
    "Superior",       # WI (Duluth-Superior) vs corpus's NE
    "Texarkana",      # TX (literal twin city across the state line) vs corpus's AR
    "York",           # PA (temporary US capital, 1777-78) vs corpus's ME
}


def generate():
    # Some raw entries above were qualified with ", <State>" purely so this
    # source file could carry two same-named towns as distinct dict keys
    # while it was being assembled (e.g. "Rochester, New York" vs.
    # "Rochester, Minnesota"). The actual model input must never contain the
    # state (that would leak the answer), so strip the qualifier here to
    # recover the bare city name actually used as input.
    normalized = [(c.split(",")[0].strip(), s) for c, s in CITY_STATE]

    # Any bare city name that resolves to more than one distinct state
    # anywhere in the corpus is a genuine disambiguation risk -> drop it
    # entirely (in addition to the names pre-flagged in AMBIGUOUS_NAMES,
    # which catches real-world-ambiguous names even when this corpus
    # happens to only encode one state for them).
    city_states = {}
    for city, state in normalized:
        city_states.setdefault(city, set()).add(state)
    cross_state_dupes = {c for c, states in city_states.items() if len(states) > 1}

    drop = AMBIGUOUS_NAMES | cross_state_dupes

    seen = {}
    for city, state in normalized:
        if city in drop:
            continue
        seen[city] = state  # consistent by construction once dupes are dropped

    records = [{"input": c, "output": s} for c, s in seen.items()]

    random.seed(42)
    random.shuffle(records)
    n = min(1000, len(records))
    dataset = records[:n]

    # self-check
    lookup = dict(seen)
    for item in dataset:
        assert lookup[item["input"]] == item["output"]

    inputs = [d["input"] for d in dataset]
    assert len(inputs) == len(set(inputs))

    from collections import Counter
    print("total candidates (after ambiguity filter):", len(records))
    state_counts = Counter(d["output"] for d in dataset)
    print("num distinct states represented:", len(state_counts))
    print("min/max per state:", min(state_counts.values()), max(state_counts.values()))

    return dataset


if __name__ == "__main__":
    dataset = generate()
    print("n =", len(dataset))
    with open("dataset_files/extended_tasks/us-city-state.json", "w") as f:
        json.dump(dataset, f, indent=2)
