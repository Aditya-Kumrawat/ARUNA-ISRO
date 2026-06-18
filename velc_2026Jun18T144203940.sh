#! /bin/bash
#Sample bash script to automate data download via PRADAN. 
#Windows users may install wget.exe and write a batch script in the same lines.
#Prequisites: Login to Pradan in your browser, select data of your interest and download script for the session
#Caution: There are session download limits, request rate limit and session timeouts in place, etc.
#	Violations may lead to blocking. Use script to ease the manual data download efforts but do not load the server.

cookies="FGTServer=03DE191863F4388C06A7AAAF7E0136FBD15060DF21FA637D82A675307CD5BF28BF8658CAFD950178CA994D;primefaces.download=null;FGTServer=03DE191863F4388C06A7AAAF7E0136FBD15060DF21FA637D82A675307CD5BF28BF8658CAFD950178CA994D;JSESSIONID=b2c304ac689ed0db9589a4a08b3d;FGTServer=03DE191863F4388C06A7AAAF7E0136FBD15060DF21FA637D82A675307CD5BF28BF8658CAFD950178CA994D;JSESSIONID=b23c2313ee431426842bff96fad8;OAuth_Token_Request_State=b110114a-67ea-4c8a-8b95-8ae499a9a0c0;"
urlPrefix="https://pradan1.issdc.gov.in"
#proxyOptions are required if your organization uses proxy to connect to Internet.
#proxyOptions="-e use_proxy=yes -e https_proxy=127.0.0.1:8080"
proxyOptions=""

#keepalive for 1 day max
counter=144;while [ $counter -gt 0 ]; do sleep 10m; wget $proxyOptions -N --content-disposition --tries=1 --no-cookies --header "Cookie: $cookies" $urlPrefix"/al1/protected/payload.xhtml"; counter=$(($counter-1)); done &
bdpid=$!

dataFilePaths=("/al1/protected/downloadData/velc/level1/2026/03/T26_0194/VS1_T26_0194_003417_20260314_135938_HG_lev1_V2_1.fits?velc" "/al1/protected/downloadData/velc/level1/2026/03/T26_0194/VS1_T26_0194_003417_20260314_135938_LG_lev1_V2_1.fits?velc" "/al1/protected/downloadData/velc/level1/2026/03/T26_0194/VS1_T26_0194_003417_20260314_135849_HG_lev1_V2_1.fits?velc" "/al1/protected/downloadData/velc/level1/2026/03/T26_0194/VS1_T26_0194_003417_20260314_135849_LG_lev1_V2_1.fits?velc" "/al1/protected/downloadData/velc/level1/2026/03/T26_0194/VS1_T26_0194_003417_20260314_135800_HG_lev1_V2_1.fits?velc" "/al1/protected/downloadData/velc/level1/2026/03/T26_0194/VS1_T26_0194_003417_20260314_135800_LG_lev1_V2_1.fits?velc" "/al1/protected/downloadData/velc/level1/2026/03/T26_0194/VS1_T26_0194_003417_20260314_135711_HG_lev1_V2_1.fits?velc" "/al1/protected/downloadData/velc/level1/2026/03/T26_0194/VS1_T26_0194_003417_20260314_135711_LG_lev1_V2_1.fits?velc" "/al1/protected/downloadData/velc/level1/2026/03/T26_0194/VS1_T26_0194_003417_20260314_135622_HG_lev1_V2_1.fits?velc" "/al1/protected/downloadData/velc/level1/2026/03/T26_0194/VS1_T26_0194_003417_20260314_135622_LG_lev1_V2_1.fits?velc" "/al1/protected/downloadData/velc/level1/2026/03/T26_0194/VS1_T26_0194_003417_20260314_135533_HG_lev1_V2_1.fits?velc" "/al1/protected/downloadData/velc/level1/2026/03/T26_0194/VS1_T26_0194_003417_20260314_135533_LG_lev1_V2_1.fits?velc" "/al1/protected/downloadData/velc/level1/2026/03/T26_0194/VS1_T26_0194_003417_20260314_135444_HG_lev1_V2_1.fits?velc" "/al1/protected/downloadData/velc/level1/2026/03/T26_0194/VS1_T26_0194_003417_20260314_135444_LG_lev1_V2_1.fits?velc" "/al1/protected/downloadData/velc/level1/2026/03/T26_0194/VS1_T26_0194_003417_20260314_135355_HG_lev1_V2_1.fits?velc" "/al1/protected/downloadData/velc/level1/2026/03/T26_0194/VS1_T26_0194_003417_20260314_135355_LG_lev1_V2_1.fits?velc" "/al1/protected/downloadData/velc/level1/2026/03/T26_0194/VS1_T26_0194_003417_20260314_135306_HG_lev1_V2_1.fits?velc" "/al1/protected/downloadData/velc/level1/2026/03/T26_0194/VS1_T26_0194_003417_20260314_135306_LG_lev1_V2_1.fits?velc" "/al1/protected/downloadData/velc/level1/2026/03/T26_0194/VS1_T26_0194_003417_20260314_135217_HG_lev1_V2_1.fits?velc" "/al1/protected/downloadData/velc/level1/2026/03/T26_0194/VS1_T26_0194_003417_20260314_135217_LG_lev1_V2_1.fits?velc" )

i=0;
for file in ${dataFilePaths[@]}
do 
	echo $file; 
	i=$(($i+1));
	wget $proxyOptions -x --max-redirect=0 --content-disposition --tries=1 --no-cookies --header "Cookie: $cookies" $urlPrefix$file;
	if [ $? -ne 0 ]; then
		echo "Error: Limits reached or session expired, terminating without downloading file $i: $file. You may login again later to download script for the new session and resume downloads." 
		kill -9 $bdpid
		exit -1;
	fi
done
echo "Your downloads($i) are complete."

kill -9 $bdpid

