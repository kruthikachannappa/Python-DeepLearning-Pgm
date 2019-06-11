class Employee:
    count = 0
    family = " "
    name = " "
    salary = 0.0
    department = " "

    def __init__(self, family, name,salary,department):
        self.family = family
        self.salary = salary
        self.name = name
        self.department = department
        Employee.count+=1

    def avg(self):
        if Employee.count > 0:
            return self.salary / self.count
        else:
            return 0

class FulltimeEmployee(Employee):
    def __init__(self, fa, na, sa, de):
        Employee.__init__(self, fa, na, sa, de)

    def display(self):
        print("Family:", self.family, "Name:", self.name, "Salary:", self.salary, "Department:", self.department)

Emp1 = Employee("f1", "Kenith", 1000, "science")
Emp2 = Employee("f2", "ben", 2000, "arts")
Emp3 = Employee("f3", "abby", 5000, "mechanical")
emp5 = FulltimeEmployee("f6", "some", 9000, "chemical")
Emp4 = FulltimeEmployee("f4", "kalki", 8000, "Textile")
print('Print Total Employees', Employee.count)
print("Average of salary",Emp3.avg())
Emp4.display()

