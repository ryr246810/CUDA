#include <ReadTxFile.hxx>


TxHierAttribSet ReadAttrib(const std::string& inputFile, 
			   const std::string& thaName)
{
  TxHierAttribSet tha(thaName);
  if(FileExists(inputFile) ){
	//cout<<inputFile<<" file exit"<<endl;
    ifstream inFile(inputFile.c_str()); 
    inFile >> tha;
  }
  // else{
	  // cout<<"file dose not exit"<<endl;
  // }
  return tha;
};



bool FileExists(string fileName) {
  FILE* spFile = fopen(fileName.c_str(), "r");
  if (!spFile) return false;

  fclose(spFile);
  return true;
};


