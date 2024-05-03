#include <TxMakerMap.h>
#include <ComboFields_Dynamic_Srcs.hxx>


void 
ComboFields_Dynamic_Srcs::
SetAttrib(const string& theWorkDir,
	  const TxHierAttribSet& theFaceBndTha)
{
  std::vector< std::string > sourceNames = theFaceBndTha.getNamesOfType("FieldSrc");

  if( sourceNames.size() ){
    std::cout << "\t Fields Sources are:";
    for(size_t i=0; i<sourceNames.size(); ++i)
      std::cout << " " << sourceNames[i];
    std::cout << std::endl;
  }
  

  Standard_Real dt = GetDelTime();

  // Add in all the sources
  for(int i=0; i<sourceNames.size(); i++){
    TxHierAttribSet attribs = theFaceBndTha.getAttrib(sourceNames[i]);
    cout<<"I am the "<<i<<endl;
    if(attribs.hasString("kind")) {
      std::string kind = attribs.getString("kind");
      ComboFields_Dynamic_SrcBase* oneNewFldSrc= TxMakerMap<ComboFields_Dynamic_SrcBase>::getNew(kind);
      //cout<<kind<<" src construct"<<endl;
	  
      if(oneNewFldSrc == 0){
        std::cout << "\t Source of kind " << kind << " not found." << std::endl;
        continue;
        }
      oneNewFldSrc->SetWorkDir(theWorkDir);// 位于ComboFields_Dynamic_SrcBase层
      oneNewFldSrc->SetDelTime(dt);// 位于DynObj层
      oneNewFldSrc->Init(GetFldsDefCntr(), EXCLUDING);// 位于FieldsBase层
      oneNewFldSrc->SetAttrib(attribs);// 声明于SetAttrib层，实现于子类
      oneNewFldSrc->Setup();// 声明于位于FieldsBase层和ComboFields_Dynamic_SrcBase层，实现于子类
      this->Append(oneNewFldSrc);
    }
  }
}
