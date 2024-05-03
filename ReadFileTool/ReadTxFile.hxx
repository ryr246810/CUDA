#ifndef _ReadTxFile_HeaderFile
#define _ReadTxFile_HeaderFile

#include <TxHierAttribSet.h>
#include <TxStreams.h>
#include <string>

TxHierAttribSet ReadAttrib(const std::string& inputFile, 
			   const std::string& thaName);

bool FileExists(string fileName); 

#endif
