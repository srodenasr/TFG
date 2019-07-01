<?xml version='1.0' encoding='UTF-8'?>
<Project Type="Project" LVVersion="18008000">
	<Item Name="My Computer" Type="My Computer">
		<Property Name="server.app.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="server.control.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="server.tcp.enabled" Type="Bool">false</Property>
		<Property Name="server.tcp.port" Type="Int">0</Property>
		<Property Name="server.tcp.serviceName" Type="Str">My Computer/VI Server</Property>
		<Property Name="server.tcp.serviceName.default" Type="Str">My Computer/VI Server</Property>
		<Property Name="server.vi.callsEnabled" Type="Bool">true</Property>
		<Property Name="server.vi.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="specify.custom.address" Type="Bool">false</Property>
		<Item Name="Controls" Type="Folder" URL="../Controls">
			<Property Name="NI.DISK" Type="Bool">true</Property>
		</Item>
		<Item Name="VIs" Type="Folder" URL="../VIs">
			<Property Name="NI.DISK" Type="Bool">true</Property>
		</Item>
		<Item Name="Initialize.vi" Type="VI" URL="../Initialize.vi"/>
		<Item Name="Test01_Utilities.vi" Type="VI" URL="../Test01_Utilities.vi"/>
		<Item Name="Test02_GPU_vs_CPU_Erode.vi" Type="VI" URL="../Test02_GPU_vs_CPU_Erode.vi"/>
		<Item Name="Test03_PIPELINE.vi" Type="VI" URL="../Test03_PIPELINE.vi"/>
		<Item Name="Dependencies" Type="Dependencies">
			<Item Name="user.lib" Type="Folder">
				<Item Name="MGI Average (DBL[]).vi" Type="VI" URL="/&lt;userlib&gt;/_MGI/1D Array/MGI Average/MGI Average (DBL[]).vi"/>
				<Item Name="MGI Average (DBL[][]).vi" Type="VI" URL="/&lt;userlib&gt;/_MGI/1D Array/MGI Average/MGI Average (DBL[][]).vi"/>
				<Item Name="MGI Average (SGL[]).vi" Type="VI" URL="/&lt;userlib&gt;/_MGI/1D Array/MGI Average/MGI Average (SGL[]).vi"/>
				<Item Name="MGI Average (SGL[][]).vi" Type="VI" URL="/&lt;userlib&gt;/_MGI/1D Array/MGI Average/MGI Average (SGL[][]).vi"/>
				<Item Name="MGI Average.vi" Type="VI" URL="/&lt;userlib&gt;/_MGI/1D Array/MGI Average.vi"/>
			</Item>
			<Item Name="vi.lib" Type="Folder">
				<Item Name="FormatTime String.vi" Type="VI" URL="/&lt;vilib&gt;/express/express execution control/ElapsedTimeBlock.llb/FormatTime String.vi"/>
				<Item Name="High Resolution Relative Seconds.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/High Resolution Relative Seconds.vi"/>
				<Item Name="Image Type" Type="VI" URL="/&lt;vilib&gt;/vision/Image Controls.llb/Image Type"/>
				<Item Name="IMAQ ArrayToImage" Type="VI" URL="/&lt;vilib&gt;/vision/Basics.llb/IMAQ ArrayToImage"/>
				<Item Name="IMAQ Create" Type="VI" URL="/&lt;vilib&gt;/vision/Basics.llb/IMAQ Create"/>
				<Item Name="IMAQ Image.ctl" Type="VI" URL="/&lt;vilib&gt;/vision/Image Controls.llb/IMAQ Image.ctl"/>
				<Item Name="IMAQ ImageToArray" Type="VI" URL="/&lt;vilib&gt;/vision/Basics.llb/IMAQ ImageToArray"/>
				<Item Name="IMAQ ReadFile 2" Type="VI" URL="/&lt;vilib&gt;/vision/Files.llb/IMAQ ReadFile 2"/>
				<Item Name="NI_Vision_Development_Module.lvlib" Type="Library" URL="/&lt;vilib&gt;/vision/NI_Vision_Development_Module.lvlib"/>
				<Item Name="subElapsedTime.vi" Type="VI" URL="/&lt;vilib&gt;/express/express execution control/ElapsedTimeBlock.llb/subElapsedTime.vi"/>
			</Item>
			<Item Name="nivision.dll" Type="Document" URL="nivision.dll">
				<Property Name="NI.PreserveRelativePath" Type="Bool">true</Property>
			</Item>
			<Item Name="nivissvc.dll" Type="Document" URL="nivissvc.dll">
				<Property Name="NI.PreserveRelativePath" Type="Bool">true</Property>
			</Item>
			<Item Name="System" Type="VI" URL="System">
				<Property Name="NI.PreserveRelativePath" Type="Bool">true</Property>
			</Item>
		</Item>
		<Item Name="Build Specifications" Type="Build">
			<Item Name="GPU_vs_CPU" Type="EXE">
				<Property Name="App_copyErrors" Type="Bool">true</Property>
				<Property Name="App_INI_aliasGUID" Type="Str">{9DF123B7-39D8-4968-8892-28D440D7814E}</Property>
				<Property Name="App_INI_GUID" Type="Str">{EE8C548D-E342-4F13-BC96-5A22713735B0}</Property>
				<Property Name="App_serverConfig.httpPort" Type="Int">8002</Property>
				<Property Name="App_winsec.description" Type="Str">http://www.AUTIS.com</Property>
				<Property Name="Bld_autoIncrement" Type="Bool">true</Property>
				<Property Name="Bld_buildCacheID" Type="Str">{7FA0FC68-27C3-4D7C-AAB8-CEE958CF9D33}</Property>
				<Property Name="Bld_buildSpecName" Type="Str">GPU_vs_CPU</Property>
				<Property Name="Bld_excludeInlineSubVIs" Type="Bool">true</Property>
				<Property Name="Bld_excludeLibraryItems" Type="Bool">true</Property>
				<Property Name="Bld_excludePolymorphicVIs" Type="Bool">true</Property>
				<Property Name="Bld_localDestDir" Type="Path">../TEST EJECUTABLES/TEST 2 - GPU vs CPU</Property>
				<Property Name="Bld_localDestDirType" Type="Str">relativeToCommon</Property>
				<Property Name="Bld_modifyLibraryFile" Type="Bool">true</Property>
				<Property Name="Bld_previewCacheID" Type="Str">{D098BA9B-170D-41AE-8F57-B490F92898BE}</Property>
				<Property Name="Bld_version.build" Type="Int">1</Property>
				<Property Name="Bld_version.major" Type="Int">1</Property>
				<Property Name="Destination[0].destName" Type="Str">GPU_vs_CPU.exe</Property>
				<Property Name="Destination[0].path" Type="Path">../TEST EJECUTABLES/TEST 2 - GPU vs CPU/GPU_vs_CPU.exe</Property>
				<Property Name="Destination[0].preserveHierarchy" Type="Bool">true</Property>
				<Property Name="Destination[0].type" Type="Str">App</Property>
				<Property Name="Destination[1].destName" Type="Str">Support Directory</Property>
				<Property Name="Destination[1].path" Type="Path">../TEST EJECUTABLES/TEST 2 - GPU vs CPU/data</Property>
				<Property Name="DestinationCount" Type="Int">2</Property>
				<Property Name="Source[0].itemID" Type="Str">{19DDDD91-04C4-488D-8F31-F52492028EDF}</Property>
				<Property Name="Source[0].type" Type="Str">Container</Property>
				<Property Name="Source[1].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[1].itemID" Type="Ref">/My Computer/Initialize.vi</Property>
				<Property Name="Source[1].sourceInclusion" Type="Str">TopLevel</Property>
				<Property Name="Source[1].type" Type="Str">VI</Property>
				<Property Name="Source[2].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[2].itemID" Type="Ref">/My Computer/Test02_GPU_vs_CPU_Erode.vi</Property>
				<Property Name="Source[2].sourceInclusion" Type="Str">TopLevel</Property>
				<Property Name="Source[2].type" Type="Str">VI</Property>
				<Property Name="SourceCount" Type="Int">3</Property>
				<Property Name="TgtF_companyName" Type="Str">AUTIS</Property>
				<Property Name="TgtF_fileDescription" Type="Str">GPU_vs_CPU</Property>
				<Property Name="TgtF_internalName" Type="Str">GPU_vs_CPU</Property>
				<Property Name="TgtF_legalCopyright" Type="Str">Copyright © 2019 AUTIS</Property>
				<Property Name="TgtF_productName" Type="Str">GPU_vs_CPU</Property>
				<Property Name="TgtF_targetfileGUID" Type="Str">{51FA8FCF-6611-4281-BC58-9D7F8F86F383}</Property>
				<Property Name="TgtF_targetfileName" Type="Str">GPU_vs_CPU.exe</Property>
				<Property Name="TgtF_versionIndependent" Type="Bool">true</Property>
			</Item>
			<Item Name="Pipeline" Type="EXE">
				<Property Name="App_copyErrors" Type="Bool">true</Property>
				<Property Name="App_INI_aliasGUID" Type="Str">{08D177EA-D48C-40B7-BF8B-D47187CF4561}</Property>
				<Property Name="App_INI_GUID" Type="Str">{CF58ED38-659B-49A0-875B-243829A3FEB2}</Property>
				<Property Name="App_serverConfig.httpPort" Type="Int">8002</Property>
				<Property Name="App_winsec.description" Type="Str">http://www.AUTIS.com</Property>
				<Property Name="Bld_autoIncrement" Type="Bool">true</Property>
				<Property Name="Bld_buildCacheID" Type="Str">{0C3E6A66-B586-4773-9E60-3769ED95E48A}</Property>
				<Property Name="Bld_buildSpecName" Type="Str">Pipeline</Property>
				<Property Name="Bld_excludeInlineSubVIs" Type="Bool">true</Property>
				<Property Name="Bld_excludeLibraryItems" Type="Bool">true</Property>
				<Property Name="Bld_excludePolymorphicVIs" Type="Bool">true</Property>
				<Property Name="Bld_localDestDir" Type="Path">../TEST EJECUTABLES/TEST 3 - PIPELINE</Property>
				<Property Name="Bld_localDestDirType" Type="Str">relativeToCommon</Property>
				<Property Name="Bld_modifyLibraryFile" Type="Bool">true</Property>
				<Property Name="Bld_previewCacheID" Type="Str">{F08A73A2-C495-4BF1-A8DE-FADB2B680D60}</Property>
				<Property Name="Bld_version.build" Type="Int">2</Property>
				<Property Name="Bld_version.major" Type="Int">1</Property>
				<Property Name="Destination[0].destName" Type="Str">Pipeline.exe</Property>
				<Property Name="Destination[0].path" Type="Path">../TEST EJECUTABLES/TEST 3 - PIPELINE/Pipeline.exe</Property>
				<Property Name="Destination[0].preserveHierarchy" Type="Bool">true</Property>
				<Property Name="Destination[0].type" Type="Str">App</Property>
				<Property Name="Destination[1].destName" Type="Str">Support Directory</Property>
				<Property Name="Destination[1].path" Type="Path">../TEST EJECUTABLES/TEST 3 - PIPELINE/data</Property>
				<Property Name="DestinationCount" Type="Int">2</Property>
				<Property Name="Source[0].itemID" Type="Str">{19DDDD91-04C4-488D-8F31-F52492028EDF}</Property>
				<Property Name="Source[0].type" Type="Str">Container</Property>
				<Property Name="Source[1].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[1].itemID" Type="Ref">/My Computer/Initialize.vi</Property>
				<Property Name="Source[1].sourceInclusion" Type="Str">TopLevel</Property>
				<Property Name="Source[1].type" Type="Str">VI</Property>
				<Property Name="Source[2].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[2].itemID" Type="Ref">/My Computer/Test03_PIPELINE.vi</Property>
				<Property Name="Source[2].sourceInclusion" Type="Str">TopLevel</Property>
				<Property Name="Source[2].type" Type="Str">VI</Property>
				<Property Name="SourceCount" Type="Int">3</Property>
				<Property Name="TgtF_companyName" Type="Str">AUTIS</Property>
				<Property Name="TgtF_fileDescription" Type="Str">Pipeline</Property>
				<Property Name="TgtF_internalName" Type="Str">Pipeline</Property>
				<Property Name="TgtF_legalCopyright" Type="Str">Copyright © 2019 AUTIS</Property>
				<Property Name="TgtF_productName" Type="Str">Pipeline</Property>
				<Property Name="TgtF_targetfileGUID" Type="Str">{E00B1D87-81BD-4F68-8419-3D51782BFBB3}</Property>
				<Property Name="TgtF_targetfileName" Type="Str">Pipeline.exe</Property>
				<Property Name="TgtF_versionIndependent" Type="Bool">true</Property>
			</Item>
			<Item Name="Utilities" Type="EXE">
				<Property Name="App_copyErrors" Type="Bool">true</Property>
				<Property Name="App_INI_aliasGUID" Type="Str">{EF136AA7-69BF-4566-9A1B-E513F8B89C34}</Property>
				<Property Name="App_INI_GUID" Type="Str">{0E1D1717-6BF5-48CC-9F09-C6836779AEF6}</Property>
				<Property Name="App_serverConfig.httpPort" Type="Int">8002</Property>
				<Property Name="App_winsec.description" Type="Str">http://www.AUTIS.com</Property>
				<Property Name="Bld_autoIncrement" Type="Bool">true</Property>
				<Property Name="Bld_buildCacheID" Type="Str">{E21D4696-E912-472D-8DD3-1E73CD64676F}</Property>
				<Property Name="Bld_buildSpecName" Type="Str">Utilities</Property>
				<Property Name="Bld_excludeInlineSubVIs" Type="Bool">true</Property>
				<Property Name="Bld_excludeLibraryItems" Type="Bool">true</Property>
				<Property Name="Bld_excludePolymorphicVIs" Type="Bool">true</Property>
				<Property Name="Bld_localDestDir" Type="Path">../TEST EJECUTABLES/TEST 1 - UTILITIES</Property>
				<Property Name="Bld_localDestDirType" Type="Str">relativeToCommon</Property>
				<Property Name="Bld_modifyLibraryFile" Type="Bool">true</Property>
				<Property Name="Bld_previewCacheID" Type="Str">{8C5339E6-EAB0-44C6-A539-663C5CD6AF84}</Property>
				<Property Name="Bld_version.build" Type="Int">1</Property>
				<Property Name="Bld_version.major" Type="Int">1</Property>
				<Property Name="Destination[0].destName" Type="Str">Utilities.exe</Property>
				<Property Name="Destination[0].path" Type="Path">../TEST EJECUTABLES/TEST 1 - UTILITIES/Utilities.exe</Property>
				<Property Name="Destination[0].preserveHierarchy" Type="Bool">true</Property>
				<Property Name="Destination[0].type" Type="Str">App</Property>
				<Property Name="Destination[1].destName" Type="Str">Support Directory</Property>
				<Property Name="Destination[1].path" Type="Path">../TEST EJECUTABLES/TEST 1 - UTILITIES/data</Property>
				<Property Name="DestinationCount" Type="Int">2</Property>
				<Property Name="Source[0].itemID" Type="Str">{19DDDD91-04C4-488D-8F31-F52492028EDF}</Property>
				<Property Name="Source[0].type" Type="Str">Container</Property>
				<Property Name="Source[1].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[1].itemID" Type="Ref">/My Computer/Initialize.vi</Property>
				<Property Name="Source[1].sourceInclusion" Type="Str">TopLevel</Property>
				<Property Name="Source[1].type" Type="Str">VI</Property>
				<Property Name="Source[2].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[2].itemID" Type="Ref">/My Computer/Test01_Utilities.vi</Property>
				<Property Name="Source[2].sourceInclusion" Type="Str">TopLevel</Property>
				<Property Name="Source[2].type" Type="Str">VI</Property>
				<Property Name="SourceCount" Type="Int">3</Property>
				<Property Name="TgtF_companyName" Type="Str">AUTIS</Property>
				<Property Name="TgtF_fileDescription" Type="Str">Utilities</Property>
				<Property Name="TgtF_internalName" Type="Str">Utilities</Property>
				<Property Name="TgtF_legalCopyright" Type="Str">Copyright © 2019 AUTIS</Property>
				<Property Name="TgtF_productName" Type="Str">Utilities</Property>
				<Property Name="TgtF_targetfileGUID" Type="Str">{52F98808-7CB7-4278-96D6-C4E882176178}</Property>
				<Property Name="TgtF_targetfileName" Type="Str">Utilities.exe</Property>
				<Property Name="TgtF_versionIndependent" Type="Bool">true</Property>
			</Item>
		</Item>
	</Item>
</Project>
