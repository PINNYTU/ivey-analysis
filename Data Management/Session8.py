# New program to calculate metric fuel usage
print("This program calculates litres per 100 km.")
km = input("Enter kilometres driven:")
km = float(km)
litres = input("Enter litres used:")
litres = float(litres)
lpk = litres / ( km / 100 )
print("Litres per 100 km:", lpk)