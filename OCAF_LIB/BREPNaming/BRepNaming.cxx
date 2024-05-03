
#include <CAGDDefine.hxx>

#include "BRepNaming.ixx"
#include <TopTools_ListOfShape.hxx>
#include <TopTools_ListIteratorOfListOfShape.hxx>
#include <TDF_Label.hxx>
#include <TDF_ChildIterator.hxx>
#include <TNaming.hxx>
#include <TNaming_Iterator.hxx>
#include <TNaming_Builder.hxx>
#include <TNaming_NamedShape.hxx>
//=======================================================================
//function : CleanStructure
//purpose  : 
//=======================================================================

void BRepNaming::CleanStructure(const TDF_Label& theLabel) {
  
  TopTools_ListOfShape Olds;
  TopTools_ListOfShape News;
  TNaming_Evolution    anEvol;
  TNaming_Iterator     anIt(theLabel);

  if (anIt.More()) {
    anEvol = anIt.Evolution();
    for ( ; anIt.More(); anIt.Next()) {
      Olds.Append(anIt.OldShape());
      News.Append(anIt.NewShape());
    }

    TopTools_ListIteratorOfListOfShape itOlds(Olds);
    TopTools_ListIteratorOfListOfShape itNews(News);
    TNaming_Builder aBuilder(theLabel);
    anEvol = TNaming_DELETE;

    for ( ;itOlds.More() ; itOlds.Next(),itNews.Next()) {
      const TopoDS_Shape& OS = itOlds.Value();
      const TopoDS_Shape& NS = itNews.Value();
      LoadNamedShape ( aBuilder, anEvol, OS, NS);
    }
  }

  for (TDF_ChildIterator chlIt(theLabel, Standard_True); chlIt.More(); chlIt.Next()) {
    CleanStructure (chlIt.Value());
  }
}

//=======================================================================
//function : LoadNamedShape
//purpose  : 
//=======================================================================

void BRepNaming::LoadNamedShape (TNaming_Builder& theBuilder,
				 const TNaming_Evolution theEvol,
				 const TopoDS_Shape& theOS,
				 const TopoDS_Shape& theNS)
{
  switch (theEvol) 
    {
    case TNaming_PRIMITIVE :
      {
	theBuilder.Generated(theNS);
	break;
      }
    case TNaming_GENERATED :
      {
	theBuilder.Generated(theOS, theNS);
	break;
      }
    case TNaming_MODIFY :
      {
	theBuilder.Modify(theOS, theNS);
	break;
      }
    case TNaming_DELETE :
      {
	theBuilder.Delete (theOS);
	break;
      }
    case TNaming_SELECTED :
      {
	theBuilder.Select(theNS, theOS);
      }
#ifndef DEB
    default:
      break;
#endif
    }
}

//=======================================================================
//function : Displace
//purpose  : if WithOld == True, dsplace with old subshapes
//=======================================================================

void BRepNaming::Displace (const TDF_Label& theLabel,
			   const TopLoc_Location& theLoc,
			   const Standard_Boolean theWithOld)
{
  TopTools_ListOfShape Olds;
  TopTools_ListOfShape News;
  TNaming_Evolution    Evol;
  TNaming_Iterator     it(theLabel);

  if (it.More()) {
    Evol = it.Evolution();
    for ( ; it.More(); it.Next()) {
      Olds.Append(it.OldShape());
      News.Append(it.NewShape());
    }

    TopTools_ListIteratorOfListOfShape itOlds(Olds);
    TopTools_ListIteratorOfListOfShape itNews(News);
    TNaming_Builder B(theLabel);

    for ( ;itOlds.More() ; itOlds.Next(),itNews.Next()) {
      TopoDS_Shape OS,NS;
      const TopoDS_Shape& SO = itOlds.Value();
      const TopoDS_Shape& SN = itNews.Value();
      OS = SO;
      if (theWithOld && !SO.IsNull()) OS = SO.Moved(theLoc);
      if (!SN.IsNull()) NS = SN.Moved(theLoc);

      LoadNamedShape ( B, Evol, OS, NS);
    }
  }
  for (TDF_ChildIterator ciL(theLabel); ciL.More(); ciL.Next()) {
    Displace (ciL.Value(),theLoc,theWithOld);
  }
}
