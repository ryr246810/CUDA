#include <Geom_TxtBuilders.hxx>

#include <TxMaker.h>
#include <TxMakerMap.h>


Geom_TxtBuilders::
Geom_TxtBuilders()
{
  m_DocCtrl = NULL;
}


Geom_TxtBuilders::
~Geom_TxtBuilders()
{
  ClearBuilders();
}


void 
Geom_TxtBuilders::
Init(OCAFDocumentCtrl* _ocafDocCtrl)
{
  m_DocCtrl = _ocafDocCtrl;
}


void
Geom_TxtBuilders::
ClearBuilders()
{
  for (Standard_Size idx = m_Builders.size(); idx>0; delete m_Builders[--idx]);
  m_Builders.clear();
}


void
Geom_TxtBuilders::
SetAttrib(const TxHierAttribSet& tas)
{
  ClearBuilders();

  std::vector< std::string > geomCtrlNames = tas.getNamesOfType("GeomCtrl");
  if(geomCtrlNames.empty()){
    cout<<"error-----------------------------------No GeomCtrl is defined"<<endl;
    exit(1);
  }else{
    if(geomCtrlNames.size()>1){
      cout<<"warning-----------------------------------GridCtrl are repeatedly defined"<<endl;
    }
  }

  TxHierAttribSet tha = tas.getAttrib(geomCtrlNames[0]);



  std::vector< std::string > builderNames = tha.getNamesOfType("GeomBuilder");


  if( builderNames.size() ){
    std::cout << "\t The All Builders are:";
    for(size_t i=0; i<builderNames.size(); ++i)
      std::cout << " " << builderNames[i];
    std::cout << std::endl;
  }
  

  // Add in all the builders
  for(size_t i=0; i<builderNames.size(); ++i){
    TxHierAttribSet attribs = tha.getAttrib(builderNames[i]);
    if(attribs.hasString("kind")) {
      std::string kind = attribs.getString("kind");

      Geom_TxtBuilderBase* oneNewBuilder= TxMakerMap<Geom_TxtBuilderBase>::getNew(kind);
      if(oneNewBuilder == 0){
	std::cout << "\t Source of kind " << kind << " not found." << std::endl;
	continue;
      }
      oneNewBuilder->SetAttrib(attribs);
      oneNewBuilder->Init(m_DocCtrl);
      oneNewBuilder->Build();
      m_Builders.push_back(oneNewBuilder);
    }
  }
}
