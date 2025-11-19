import os.path

class Flujo:
    ruta = ""
    flujo = ""
    indiceFlujo = 0
    contadorLinea = 0
    contadorCaracter = 0
    caracteresEnLinea = []

    def __init__(self, ruta):
        """
        Inicializa una instancia del flujo a partir de un archivo de texto.

        Args:
            ruta (str): Ruta del archivo a leer.

        Raises:
            FileNotFoundError: Si el archivo no existe en la ruta proporcionada.
        """
        if not os.path.isfile(ruta):
            raise FileNotFoundError(f'El archivo {ruta} no existe.')

        self.ruta = ruta
        archivo = open(ruta, 'r')
        self.flujo = archivo.read()
        archivo.close()

        self.indiceFlujo = 0
        self.contadorLinea = 0
        self.contadorCaracter = 0
        self.caracteresEnLinea = []

    def NewCar(self):
        """
        Devuelve el siguiente carácter del flujo.

        Returns:
            str | None | bool:
                - str: El siguiente carácter del flujo.
                - EOF: Si se alcanzó el final del flujo (EOF).
                - False: Si ya no quedan más caracteres después del EOF.
        """
        if self.indiceFlujo == len(self.flujo):  # Si hemos llegado al final del flujo EOF
            self.contadorCaracter += 1
            self.indiceFlujo += 1
            caracter = "EOF"
        elif self.indiceFlujo > len(self.flujo):  # Si ya hemos devuelto el EOF
            caracter = False
        else:
            caracter = self.flujo[self.indiceFlujo]
            self.indiceFlujo += 1
            if caracter == '\n':
                self.contadorLinea += 1
                self.caracteresEnLinea.append(self.contadorCaracter)
                self.contadorCaracter = 0
            else:
                self.contadorCaracter += 1

        return caracter

    def Devolver(self):
        """
        Retrocede una posición en el flujo.

        Returns:
            bool: True si el retroceso fue exitoso, False si ya se está al inicio del flujo.
        """
        if self.indiceFlujo <= 0:
            return False

        self.indiceFlujo -= 1
        if self.contadorCaracter > 0:
            self.contadorCaracter -= 1
        else:
            self.contadorLinea -= 1
            self.contadorCaracter = self.caracteresEnLinea.pop()

        return True

    def NumLinea(self):
        """
        Obtiene el número de línea actual en el flujo.

        Returns:
            int: Número de línea (0 si es la primera).
        """
        if self.contadorCaracter <= 0 and self.contadorLinea > 0:
            return (self.contadorLinea - 1) + 1
        return self.contadorLinea + 1

    def NumCaracter(self):
        """
        Obtiene la posición del último carácter leído en la línea actual.

        Returns:
            int:
                - -1 si no se ha leído ningún carácter.
                - 0 si es el primer carácter de la línea.
                - n si es la posición n de la línea actual.
        """
        if self.contadorCaracter <= 0:
            if len(self.caracteresEnLinea) == 0:
                return -1
            return self.caracteresEnLinea[-1]
        return self.contadorCaracter - 1
    
    # Funciones adicionales al ejercicio

    def DevolverN(self, n):
        """
        Retrocede n posiciones en el flujo.

        Args:
            n (int): Número de posiciones a retroceder.

        Returns:
            int: Número de posiciones que se lograron retroceder efectivamente.
        """
        n = abs(n)
        posiciones = 0

        if n > self.indiceFlujo:
            n = self.indiceFlujo

        for _ in range(n):
            if self.Devolver():
                posiciones += 1
        return posiciones

    def AvanzarN(self, n):
        """
        Avanza n posiciones en el flujo.

        Args:
            n (int): Número de posiciones a avanzar.

        Returns:
            str: Cadena resultante de los caracteres avanzados.
        """
        n = abs(n)
        if n > len(self.flujo) - self.indiceFlujo:
            n = len(self.flujo) - self.indiceFlujo

        cadena = []
        for _ in range(n):
            cadena.append(self.NewCar())
        return ''.join(cadena)
    
    def SiguienteLinea(self,n = 1):
        n = abs(n)
        if n > len(self.flujo) - self.indiceFlujo:
            n = len(self.flujo) - self.indiceFlujo

        cadena = []
        char = None
        while(char != '\n' and char != "EOF"):
            char = self.NewCar()
            cadena.append(char)
        return cadena    

    def Reasignar(self, ruta):
        """
        Reasigna el flujo para leer desde otro archivo.

        Args:
            ruta (str): Ruta del nuevo archivo a leer.
        """
        self.__init__(ruta)

    def __str__(self):
        """
        Representación en cadena del flujo.

        Returns:
            str: Ruta del archivo y contenido del flujo.
        """
        return f'Ruta={self.ruta}\nFlujo:\n{self.flujo}'



#pruebas()