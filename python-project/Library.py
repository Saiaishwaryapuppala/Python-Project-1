class Library:
    book_list = []

    def __init__(self, listofbooks):
        Library.book_list += listofbooks
        
    @staticmethod
    def Get_booklist():
        return Library.book_list

    @staticmethod
    def display_books():
        print("The books available in library are:- ", Library.book_list)
        print("\n")

    @staticmethod
    def checkout_books(name):
        Library.book_list.remove(name)
    
    @staticmethod
    def checkin_books(name):
        Library.book_list.append(name)

class LoanBooks(Library): 
    def __init__(self,_listofbooks):
        super().__init__(_listofbooks) # using the super call and accessing the parent class methods

class Student(Library): # Inherting the library class for checkout and check in the books
    def __init__(self, name, student_id, phonenumber, department, email):
        self.name = name
        self.__phone_number = phonenumber # private member
        self.student_id = student_id
        self.department = department
        self.email = email
        self.student_booklist = []
        
    def student_info(self):
        print("Student Info :--")
        print("Name", self.name)
        print("phonenumber", self.__phone_number)
        print("student_id", self.student_id)
        print("department", self.department)
        print("email", self.email)

    def request_book(self,name):
        _booklist= Library.Get_booklist()
        if name in _booklist:
            self.student_booklist.append(name)
            Library.checkout_books(name)
            print("Book is issued successfully")
            print("\n")
        else:
            print("Book is not available")

    def return_book(self,name):
        self.student_booklist.remove(name)
        Library.checkin_books(name)
        print("Book is returned Successfully")
        print("\n")
    
    def studentbooklist(self):
        print("Book student cart are ", self.student_booklist)
        print("\n")
        

            
        
# Stroring book in library
Library(["harry poter", "secret seven", "A Prisoner of Birth","The Sins of the Father"])
# Giving the student details
student = Student("Harish", 37, 90000000000, "cs", "harish@gmail.com")

student.display_books()
# Checkout the book to the student
student.request_book("A Prisoner of Birth")
# Display books in library
student.display_books()
# Show the student booklist
student.studentbooklist()
# Return the Book
student.return_book("A Prisoner of Birth")
student.display_books()
# Display the student Info
student.student_info()
# Loan the books from other library
temp = LoanBooks(["Heads You Win"])
temp.display_books()
# Accessing of the private member of a class
print("Accessing of the private member of a class")
student._Student__phone_number # correct way
# student.__phone_number # Wrong way to access you will get an error