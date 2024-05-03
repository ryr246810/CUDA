
#ifndef _OCAF_SolverEx_HeaderFile
#define _OCAF_SolverEx_HeaderFile

#ifndef _Standard_Boolean_HeaderFile
#include <Standard_Boolean.hxx>
#endif

#ifndef _TDF_Label_HeaderFile
#include <TDF_Label.hxx>
#endif

#ifndef _TDF_AttributeList_HeaderFile
#include <TDF_AttributeList.hxx>
#endif

#ifndef _TDF_AttributeMap_HeaderFile
#include <TDF_AttributeMap.hxx>
#endif


#ifndef _Standard_Boolean_HeaderFile
#include <Standard_Boolean.hxx>
#endif

#ifndef _TDataStd_TreeNode_HeaderFile
#include <TDataStd_TreeNode.hxx>
#endif

#ifndef _TFunction_Function_HeaderFile
#include <TFunction_Function.hxx>
#endif

#include <TDF_Tool.hxx>
#include <TDF_Reference.hxx>

//*
#ifndef _TFunction_Logbook_HeaderFile
#include <TFunction_Logbook.hxx>
#endif
//*/

#ifndef _Standard_HeaderFile
#include <Standard.hxx>
#endif

class TFunction_Logbook;


class OCAF_SolverEx  
{

public:

    inline void* operator new(size_t,void* anAddress) 
      {
        return anAddress;
      }
    inline void* operator new(size_t size) 
      { 
        return Standard::Allocate(size); 
      }
    inline void  operator delete(void *anAddress) 
      { 
        if (anAddress) Standard::Free((Standard_Address&)anAddress); 
      }

 // Methods PUBLIC
 // 
  Standard_EXPORT  OCAF_SolverEx();
  Standard_EXPORT  Standard_Boolean Solve(const TDF_Label& theAccessLable) ;
  Standard_EXPORT  Standard_Boolean SolveFrom(const Handle(TDataStd_TreeNode)& theNode) ;

  Standard_EXPORT  Standard_Boolean IsCyclicLink(const TDF_Label& theFrom,
						 const TDF_Label& theTo) ;

  Standard_EXPORT  Standard_Boolean ComputeFunction(const Handle(TFunction_Function)& theFunction,
						    TDF_AttributeMap& theSolved);

  Standard_EXPORT  void GetAttachments(const Handle(TFunction_Function)& theFunction, 
				       TDF_AttributeMap& theMap) const;

  Standard_EXPORT  void ComputeExecutionList(const TDF_Label& theAccessLabel);

  Standard_EXPORT  const TDF_AttributeList& GetExecutionList() const;
  
  Standard_EXPORT  void Dump() const;

protected:

 // Methods PROTECTED
 // 


 // Fields PROTECTED
 //


private: 
  
 // Methods PRIVATE
 // 
  Standard_EXPORT   Standard_Boolean PerformSolving(Handle(TFunction_Logbook)& theLogbook,
						    const Handle(TFunction_Function)& theSkipFunction,
						    const Standard_Boolean theWithCheck) const;
  

 // Fields PRIVATE
 //
  TDF_AttributeList myList;


};





// other inline functions and methods (like "C++: function call" methods)
//


#endif
