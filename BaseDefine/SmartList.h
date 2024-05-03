/*
Definition for template SmartListItem,SmartList,SmartListIter & macro ApplyToList(function,theList,listType)
*/
#ifndef SmartList_Headerfile
#define SmartList_Headerfile


/////////////////////////////////////////////////////////////////////////////////////////
template <class Type> class SmartListItem
{
public:
  SmartListItem(Type *newitem, SmartListItem<Type> *nxt, const char* key = ""){
    data=newitem;
    next=nxt;
    itemName = _strdup(key);
  }
  ~SmartListItem(){
    if (itemName) delete itemName;
    itemName = NULL;
    next = NULL;
    data = NULL;
  }
  char* itemName; 					// jbha
  SmartListItem<Type> *next;
  Type *data;
};
/////////////////////////////////////////////////////////////////////////////////////////
template <class Type> class SmartList
{
 public:
  int nitems;
  SmartListItem<Type> *head;
  SmartList() {
    head=0;
    nitems=0;
  }
  SmartList(SmartList<Type> &other){
    head=0;
    nitems=0;
    *this=other;
  }
  ~SmartList(){
    removeAll();
  }

  void removeItem( Type *anItem);
  void removeAll();
  void deleteAll();
  void add(Type *newitem);
  void addToEnd(Type *newitem);  // expensive walk-the-list-to-end-and-add
  
  SmartList<Type>& operator=(SmartList<Type> &other){
    //Not a terribly efficient routine.
    //first copy of list is reversed.
    //reversed_list goes out of scope and is deleted.
    SmartListIter<Type> walk(other);
    SmartList<Type> reversed_list;
    
    removeAll();
    while(!walk.Done()) {
      reversed_list.add(walk.current());
      walk++;
    }
    SmartListIter<Type> walk2(reversed_list);
    while(!walk2.Done()){
      add(walk2.current());
      walk2++;
    }
    return *this;
  };
  
  Type * pop(void){
    if(head){
      Type* retval=head->data;
      SmartListItem<Type> *to_delete=head;
      head = head->next;
      delete to_delete;
      nitems--;
      return retval;
    }
    else
      return 0;
  }
  
  int isEmpty() {return (head==0);}
  void push(Type *newitem) { add(newitem); };
  int nItems() {return nitems;};
};

template <class Type> void SmartList<Type>::add(Type *newitem)
{
  SmartListItem<Type> *New = new SmartListItem<Type>(newitem, head);
  nitems++;
  head = New;
}

// Add an element to the end of the list.  Requires a list traversal.
template <class Type> void SmartList<Type>::addToEnd(Type *newitem)
{
  SmartListItem<Type> *New = new SmartListItem<Type>(newitem,NULL);
  SmartListItem<Type> *walk;
  nitems++;
  if(head==NULL){
    head = New;
    return;
  }
  
  // find the end of the list
  for( walk=head; walk->next!=NULL; walk=walk->next );
  walk->next = New; 
}

// use when you need to find the item first, else use version in SmartListIter
template <class Type> void SmartList<Type>::removeItem( Type *anItem )
{
  SmartListItem<Type>	*prevPtr, *currentPtr;
  
  if (head == NULL){
    return;
  }else if (head->data == anItem) {
    currentPtr = head->next;
    delete head;
    head = currentPtr;
    nitems--;
  }else{
    prevPtr = head;
    currentPtr = prevPtr->next;
    while( !(currentPtr == NULL) ){
      if ( currentPtr->data == anItem ){
	prevPtr->next = currentPtr->next;
	delete currentPtr;
	currentPtr = prevPtr->next;
	nitems--;
      }else{
	prevPtr = currentPtr;
	currentPtr = currentPtr->next;
      }
    }
  }
}

// removeAll: removes all list elements, but NOT the data!

template <class Type> void SmartList<Type>::removeAll()
{
  SmartListItem<Type> *next;
  while (head){
    next = head->next;
    delete head;
    head = next;
  }
  nitems = 0;
}

// deleteAll: removes all list elements AND data

template <class Type> void SmartList<Type>::deleteAll()
{
  SmartListItem<Type> *next;
  while ( head ){
    next = head->next;
    delete head->data;
    delete head;
    head = next;
  }
  nitems = 0;
}

/////////////////////////////////////////////////////////////////////////////////////////
template <class Type> class SmartListIter
{
 public:
  SmartListItem<Type>* currentPtr;
  SmartListItem<Type>* previousPtr;
  SmartList<Type>* theList;
  SmartListIter(){}						// empty constructor, must be used with restart
  SmartListIter(SmartList<Type>& aList){
    theList=&aList;
    restart();
  };
  void	restart() {previousPtr = currentPtr = theList->head;};
  void	restart(SmartList<Type>& aList){
    theList=&aList;
    restart();
  };
  void	restart(Type& aGroup);
  Type*	data(){
    if (currentPtr) return currentPtr->data;
    else return NULL;
  };
  Type*	current() {return data();};
  Type*	operator () () {return data();};
  void	operator++(int){
    if (currentPtr){
      previousPtr = currentPtr;
      currentPtr = currentPtr->next;
    } else restart();
  };
  int	Done() {return currentPtr==NULL;};
  int	isEmpty() {return !(theList->head==0);};
  void  deleteCurrent();
};


//	This function removes an item from the list.  Might be cleaner
//	if it is inside SmartList<Type>, but do not want order n search
//	for element.
template <class Type> void SmartListIter<Type>::deleteCurrent()
{
  if (currentPtr == NULL) return;		//	bail out if empty or at end

  if (currentPtr == theList->head){             //	if at head of list, point head to next item
    theList->head = currentPtr->next;
    previousPtr = NULL;
  }
  else previousPtr->next = currentPtr->next;
  delete	current();						//	delete the data
  delete	currentPtr;						//	delete the list item
  currentPtr = previousPtr;
  theList->nitems--;
}

/////////////////////////////////////////////////////////////////////////////////////////
#define ApplyToList(fun,Lst,LType) { SmartListIter<LType> SLI(Lst);  for(SLI.restart();!SLI.Done();SLI++) SLI.current()->fun; }
#endif
