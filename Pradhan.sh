#! /bin/bash
#Sample bash script to automate data download via PRADAN. 
#Windows users may install wget.exe and write a batch script in the same lines.
#Prequisites: Login to Pradan in your browser, select data of your interest and download script for the session
#Caution: There are session download limits, request rate limit and session timeouts in place, etc.
#	Violations may lead to blocking. Use script to ease the manual data download efforts but do not load the server.

cookies="primefaces.download=null;FGTServer=03DE191863F4388C06A7AAAF7E0136FBD15060DF21FA637D82A675307CD5BF28BF8658CAFD950178C8994D;FGTServer=03DE191863F4388C06A7AAAF7E0136FBD15060DF21FA637D82A675307CD5BF28BF8658CAFD950178C8994D;JSESSIONID=e884e40e976550e831e8002616ea;FGTServer=03DE191863F4388C06A7AAAF7E0136FBD15060DF21FA637D82A675307CD5BF28BF8658CAFD950178C8994D;OAuth_Token_Request_State=026184a6-4898-4bcd-95df-8b1cc50f233b;JSESSIONID=e886bc7bacea80c77c0f9b48c84f;"
urlPrefix="https://pradan1.issdc.gov.in"
#proxyOptions are required if your organization uses proxy to connect to Internet.
#proxyOptions="-e use_proxy=yes -e https_proxy=127.0.0.1:8080"
proxyOptions=""

#keepalive for 1 day max
counter=144;while [ $counter -gt 0 ]; do sleep 10m; wget $proxyOptions -N --content-disposition --tries=1 --no-cookies --header "Cookie: $cookies" $urlPrefix"/al1/protected/payload.xhtml"; counter=$(($counter-1)); done &
bdpid=$!

dataFilePaths=("/al1/protected/downloadData/velc/level2/2026/03/T26_0200/VS1_T26_0200_003423_20260315_232446_LG_lev2_V2_1.fits?all" "/al1/protected/downloadData/velc/level2/2026/03/T26_0200/VS1_T26_0200_003423_20260315_232446_HG_lev2_V2_1.fits?all" "/al1/protected/downloadData/velc/level2/2026/03/T26_0200/VS1_T26_0200_003423_20260315_232426_LG_lev2_V2_1.fits?all" "/al1/protected/downloadData/velc/level2/2026/03/T26_0200/VS1_T26_0200_003423_20260315_232426_HG_lev2_V2_1.fits?all" "/al1/protected/downloadData/velc/level2/2026/03/T26_0200/VS1_T26_0200_003423_20260315_232406_LG_lev2_V2_1.fits?all" "/al1/protected/downloadData/velc/level2/2026/03/T26_0200/VS1_T26_0200_003423_20260315_232406_HG_lev2_V2_1.fits?all" "/al1/protected/downloadData/velc/level2/2026/03/T26_0200/VS1_T26_0200_003423_20260315_232345_LG_lev2_V2_1.fits?all" "/al1/protected/downloadData/velc/level2/2026/03/T26_0200/VS1_T26_0200_003423_20260315_232345_HG_lev2_V2_1.fits?all" "/al1/protected/downloadData/velc/level2/2026/03/T26_0200/VS1_T26_0200_003423_20260315_232325_LG_lev2_V2_1.fits?all" "/al1/protected/downloadData/velc/level2/2026/03/T26_0200/VS1_T26_0200_003423_20260315_232325_HG_lev2_V2_1.fits?all" )

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

