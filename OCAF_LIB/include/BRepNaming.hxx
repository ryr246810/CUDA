
#ifndef _BRepNaming_HeaderFile
#define _BRepNaming_HeaderFile

#ifndef _TNaming_Evolution_HeaderFile
#include <TNaming_Evolution.hxx>
#endif
#ifndef _Standard_Boolean_HeaderFile
#include <Standard_Boolean.hxx>
#endif
class TDF_Label;
class TNaming_Builder;
class TopoDS_Shape;
class TopLoc_Location;
class BRepNaming_LoaderParent;
class BRepNaming_Loader;
class BRepNaming_TopNaming;
class BRepNaming_Box;
class BRepNaming_Prism;
class BRepNaming_Revol;
class BRepNaming_Cone;
class BRepNaming_Cylinder;
class BRepNaming_Sphere;
class BRepNaming_Torus;
class BRepNaming_CircularPipe;
class BRepNaming_Pipe;
class BRepNaming_PipeShell;
class BRepNaming_ThruSections;
class BRepNaming_BooleanOperation;
class BRepNaming_BooleanOperationFeat;
class BRepNaming_Common;
class BRepNaming_Cut;
class BRepNaming_Fuse;
class BRepNaming_Section;
class BRepNaming_Fillet;
class BRepNaming_Chamfer;
class BRepNaming_ImportShape;
class BRepNaming_Draft;
class BRepNaming_ThickSolid;
class BRepNaming_Vary;
class BRepNaming_DPrism;
class BRepNaming_Hole;
class BRepNaming_LinearForm;
class BRepNaming_Mirror;
class BRepNaming_PipeFeat;
class BRepNaming_Scale;
class BRepNaming_PrismFeat;
class BRepNaming_RevolFeat;


#ifndef _Standard_HeaderFile
#include <Standard.hxx>
#endif
#ifndef _Standard_Macro_HeaderFile
#include <Standard_Macro.hxx>
#endif

class BRepNaming  {
  
public:
  
  void* operator new(size_t,void* anAddress) {
    return anAddress;
  }
  void* operator new(size_t size)  { 
    return Standard::Allocate(size); 
  }
  void  operator delete(void *anAddress)  { 
    if (anAddress) Standard::Free((Standard_Address&)anAddress); 
  }
  // Methods PUBLIC
 // 
  Standard_EXPORT static  void CleanStructure(const TDF_Label& theLabel) ;
  Standard_EXPORT static  void LoadNamedShape(TNaming_Builder& theBuilder,const TNaming_Evolution theEvolution,const TopoDS_Shape& theOldShape,const TopoDS_Shape& theNewShape) ;
  Standard_EXPORT static  void Displace(const TDF_Label& theLabel,const TopLoc_Location& theLoc,const Standard_Boolean theWithOld = Standard_False) ;
  




protected:

 // Methods PROTECTED
 // 


 // Fields PROTECTED
 //


private: 

 // Methods PRIVATE
 // 


 // Fields PRIVATE
 //

friend class BRepNaming_LoaderParent;
friend class BRepNaming_Loader;
friend class BRepNaming_TopNaming;
friend class BRepNaming_Box;
friend class BRepNaming_Prism;
friend class BRepNaming_Revol;
friend class BRepNaming_Cone;
friend class BRepNaming_Cylinder;
friend class BRepNaming_Sphere;
friend class BRepNaming_Torus;
friend class BRepNaming_CircularPipe;
friend class BRepNaming_Pipe;
friend class BRepNaming_PipeShell;
friend class BRepNaming_ThruSections;
friend class BRepNaming_BooleanOperation;
friend class BRepNaming_BooleanOperationFeat;
friend class BRepNaming_Common;
friend class BRepNaming_Cut;
friend class BRepNaming_Fuse;
friend class BRepNaming_Section;
friend class BRepNaming_Fillet;
friend class BRepNaming_Chamfer;
friend class BRepNaming_ImportShape;
friend class BRepNaming_Draft;
friend class BRepNaming_ThickSolid;
friend class BRepNaming_Vary;
friend class BRepNaming_DPrism;
friend class BRepNaming_Hole;
friend class BRepNaming_LinearForm;
friend class BRepNaming_Mirror;
friend class BRepNaming_PipeFeat;
friend class BRepNaming_Scale;
friend class BRepNaming_PrismFeat;
friend class BRepNaming_RevolFeat;

};





// other Inline functions and methods (like "C++: function call" methods)
//


#endif
