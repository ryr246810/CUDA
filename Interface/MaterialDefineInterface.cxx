#include <MaterialDefineInterface.hxx>

#include <BaseDataDefine.hxx>
#include <OCAF_ColorMap.hxx>
#include <iostream>
//#include <config.h>

void SetupColorMap()
{
  cout<<"---------------------SetupColorMap-----1"<<endl;
  ClearColorMap();
  Standard_Boolean IsOk = Standard_False;
  OCAF_ColorMap::InsertColorDefine(1,"PEC", PEC,Quantity_NOC_GOLDENROD, IsOk);
  //OCAF_ColorMap::InsertColorDefine(2,"FREESPACE", EMFREESPACE, Quantity_NOC_BLUE1, IsOk);
  //OCAF_ColorMap::InsertColorDefine(2,"FREESPACE", EMFREESPACE, Quantity_NOC_CORNFLOWERBLUE, IsOk);
  OCAF_ColorMap::InsertColorDefine(2,"FREESPACE", EMFREESPACE, Quantity_NOC_DEEPSKYBLUE2, IsOk);

  //OCAF_ColorMap::InsertColorDefine(3,"USERDEFINED", USERDEFINED, Quantity_NOC_SEAGREEN, IsOk);
  OCAF_ColorMap::InsertColorDefine(3,"USERDEFINED", USERDEFINED, Quantity_NOC_FORESTGREEN, IsOk);

  OCAF_ColorMap::InsertColorDefine(4,"OPENPORT", OPENPORT, Quantity_NOC_RED, IsOk);
  OCAF_ColorMap::InsertColorDefine(5,"INPUTPORT", INPUTPORT, Quantity_NOC_FIREBRICK4, IsOk);
  OCAF_ColorMap::InsertColorDefine(6,"EMITTER", EMITTER0, Quantity_NOC_CHOCOLATE4, IsOk);
  OCAF_ColorMap::InsertColorDefine(7,"MURPORT", MURPORT, Quantity_NOC_CHOCOLATE3, IsOk);
  OCAF_ColorMap::InsertColorDefine(8,"PECPORT", PECPORT, Quantity_NOC_CHOCOLATE2, IsOk);
  OCAF_ColorMap::InsertColorDefine(9,"INPUTMURPORT", INPUTMURPORT, Quantity_NOC_CHOCOLATE2, IsOk);
  
  cout<<"---------------------SetupColorMap-----2"<<endl;
}


void ClearColorMap()
{
  OCAF_ColorMap::ClearColorDefine();
}




  //Quantity_NOC_CORNFLOWERBLUE  //background
  //Quantity_NOC_CHOCOLATE
  //Quantity_NOC_DARKSLATEGRAY
  //Quantity_NOC_FIREBRICK
  //Quantity_NOC_FORESTGREEN
